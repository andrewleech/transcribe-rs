//! Core ONNX inference for Qwen3-ASR: encoder, decoder prefill, and autoregressive decode.

use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use super::config::Qwen3AsrConfig;
use super::mel::log_mel_spectrogram;
use super::prompt::{build_prompt_ids, get_audio_pad_range, get_feat_extract_output_lengths};
use super::tokenizer::Qwen3Tokenizer;

#[derive(thiserror::Error, Debug)]
pub enum Qwen3Error {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Config error: {0}")]
    Config(String),
    #[error("Decode error: {0}")]
    Decode(String),
}

/// Whether to use quantized (int8) model files.
#[derive(Debug, Clone, Default)]
pub struct Qwen3ModelOptions {
    pub quantized: bool,
}

pub struct Qwen3AsrModel {
    encoder: Session,
    decoder_init: Session,
    decoder_step: Session,
    embed_tokens: Array2<f32>,
    config: Qwen3AsrConfig,
    tokenizer: Qwen3Tokenizer,
}

impl Qwen3AsrModel {
    pub fn new(
        model_dir: &Path,
        options: &Qwen3ModelOptions,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = Qwen3AsrConfig::load(model_dir)?;
        let tokenizer = Qwen3Tokenizer::new(model_dir)?;

        // Load ONNX sessions
        let encoder_path = Self::find_model_file(model_dir, "encoder", options.quantized)?;
        let decoder_init_path =
            Self::find_model_file(model_dir, "decoder_init", options.quantized)?;
        let decoder_step_path =
            Self::find_model_file(model_dir, "decoder_step", options.quantized)?;

        log::info!("Loading Qwen3-ASR encoder from {:?}", encoder_path);
        let encoder = Self::init_session(&encoder_path)?;

        log::info!(
            "Loading Qwen3-ASR decoder_init from {:?}",
            decoder_init_path
        );
        let decoder_init = Self::init_session(&decoder_init_path)?;

        log::info!(
            "Loading Qwen3-ASR decoder_step from {:?}",
            decoder_step_path
        );
        let decoder_step = Self::init_session(&decoder_step_path)?;

        // Load embedding matrix from raw binary
        let embed_path = model_dir.join("embed_tokens.bin");
        if !embed_path.exists() {
            return Err(Box::new(Qwen3Error::ModelNotFound(
                embed_path.display().to_string(),
            )));
        }
        log::info!("Loading embedding matrix from {:?}", embed_path);
        let embed_tokens = Self::load_embed_tokens(&embed_path, &config)?;
        log::info!("Embedding matrix shape: {:?}", embed_tokens.shape());

        Ok(Self {
            encoder,
            decoder_init,
            decoder_step,
            embed_tokens,
            config,
            tokenizer,
        })
    }

    fn find_model_file(
        model_dir: &Path,
        name: &str,
        quantized: bool,
    ) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        if quantized {
            let q_path = model_dir.join(format!("{}.int8.onnx", name));
            if q_path.exists() {
                return Ok(q_path);
            }
            log::warn!(
                "Quantized model {}.int8.onnx not found, falling back to {}.onnx",
                name,
                name
            );
        }
        let path = model_dir.join(format!("{}.onnx", name));
        if path.exists() {
            Ok(path)
        } else {
            Err(Box::new(Qwen3Error::ModelNotFound(
                path.display().to_string(),
            )))
        }
    }

    fn init_session(path: &Path) -> Result<Session, Qwen3Error> {
        let providers = vec![CPUExecutionProvider::default().build()];
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .commit_from_file(path)?;

        for input in session.inputs() {
            log::debug!("  Input: name={}, type={:?}", input.name(), input.dtype());
        }
        Ok(session)
    }

    fn load_embed_tokens(
        path: &Path,
        config: &Qwen3AsrConfig,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let [vocab_size, hidden_size] = config.embed_tokens_shape;
        let expected_bytes = vocab_size * hidden_size * 4; // f32
        if data.len() != expected_bytes {
            return Err(Box::new(Qwen3Error::Config(format!(
                "embed_tokens.bin size {} != expected {} ({}x{}x4)",
                data.len(),
                expected_bytes,
                vocab_size,
                hidden_size
            ))));
        }

        // Safety: reinterpret bytes as f32 slice
        let float_data: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Array2::from_shape_vec(
            (vocab_size, hidden_size),
            float_data,
        )?)
    }

    /// Run the encoder on a mel spectrogram.
    fn encode(&mut self, mel: &Array3<f32>) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        let mel_dyn = mel.clone().into_dyn();
        let inputs = ort::inputs![
            "mel" => TensorRef::from_array_view(mel_dyn.view())?,
        ];
        let outputs = self.encoder.run(inputs)?;

        let features = outputs
            .get("audio_features")
            .ok_or_else(|| Qwen3Error::Decode("Missing 'audio_features' output".into()))?
            .try_extract_array::<f32>()?;

        let shape = features.shape();
        Ok(Array3::from_shape_vec(
            (shape[0], shape[1], shape[2]),
            features.iter().cloned().collect(),
        )?)
    }

    /// Build input embeddings: embed all prompt tokens, then replace audio_pad
    /// positions with encoder output features.
    fn build_input_embeds(&self, prompt_ids: &[i64], audio_features: &Array3<f32>) -> Array3<f32> {
        let hidden_size = self.config.decoder.hidden_size;
        let seq_len = prompt_ids.len();

        // Embed all tokens
        let mut embeds = Array2::<f32>::zeros((seq_len, hidden_size));
        for (i, &id) in prompt_ids.iter().enumerate() {
            let id = id as usize;
            if id < self.embed_tokens.nrows() {
                embeds.row_mut(i).assign(&self.embed_tokens.row(id));
            }
        }

        // Replace audio_pad positions with encoder output
        let (audio_start, audio_end) =
            get_audio_pad_range(prompt_ids, self.config.special_tokens.audio_pad_token_id);
        let audio_len = audio_end - audio_start;
        for i in 0..audio_len {
            embeds
                .row_mut(audio_start + i)
                .assign(&audio_features.slice(ndarray::s![0, i, ..]));
        }

        // Reshape to [1, seq_len, hidden_size]
        embeds
            .into_shape_with_order((1, seq_len, hidden_size))
            .unwrap()
    }

    /// Run greedy decoding on audio features, returning decoded text.
    pub fn greedy_decode(
        &mut self,
        audio_features: &Array3<f32>,
        max_tokens: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let audio_token_count = audio_features.shape()[1];
        let prompt_ids = build_prompt_ids(&self.config.special_tokens, audio_token_count);
        let input_embeds = self.build_input_embeds(&prompt_ids, audio_features);
        let seq_len = prompt_ids.len();

        // Position IDs: [1, seq_len]
        let position_ids = Array2::<i64>::from_shape_fn((1, seq_len), |(_, j)| j as i64);

        // Prefill: run decoder_init
        let embeds_dyn = input_embeds.into_dyn();
        let pos_dyn = position_ids.into_dyn();

        let init_inputs = ort::inputs![
            "input_embeds" => TensorRef::from_array_view(embeds_dyn.view())?,
            "position_ids" => TensorRef::from_array_view(pos_dyn.view())?,
        ];
        let init_outputs = self.decoder_init.run(init_inputs)?;

        let logits = init_outputs
            .get("logits")
            .ok_or_else(|| Qwen3Error::Decode("Missing 'logits' from decoder_init".into()))?
            .try_extract_array::<f32>()?;
        let present_keys = init_outputs
            .get("present_keys")
            .ok_or_else(|| Qwen3Error::Decode("Missing 'present_keys'".into()))?
            .try_extract_array::<f32>()?;
        let present_values = init_outputs
            .get("present_values")
            .ok_or_else(|| Qwen3Error::Decode("Missing 'present_values'".into()))?
            .try_extract_array::<f32>()?;

        // First token from prefill logits (last position)
        let logits_shape = logits.shape();
        let last_pos = logits_shape[1] - 1;
        let vocab_size = logits_shape[2];
        let next_token = argmax_slice(&logits, last_pos, vocab_size);
        let mut output_tokens = vec![next_token];

        if self
            .config
            .special_tokens
            .eos_token_ids
            .contains(&next_token)
        {
            return Ok(self.tokenizer.decode(&output_tokens));
        }

        // Convert KV cache to owned dynamic arrays for the step loop
        let mut keys = present_keys.to_owned().into_dyn();
        let mut values = present_values.to_owned().into_dyn();
        let mut pos = seq_len as i64;

        // Autoregressive loop
        for _ in 1..max_tokens {
            let token_id = *output_tokens.last().unwrap();
            let hidden_size = self.config.decoder.hidden_size;

            // Embed the new token: [1, 1, hidden_size]
            let token_embed = {
                let row = self.embed_tokens.row(token_id as usize);
                let mut arr = Array3::<f32>::zeros((1, 1, hidden_size));
                arr.slice_mut(ndarray::s![0, 0, ..]).assign(&row);
                arr.into_dyn()
            };

            // Position: [1, 1]
            let step_pos = ArrayD::<i64>::from_shape_vec(IxDyn(&[1, 1]), vec![pos])?;

            let step_inputs = ort::inputs![
                "input_embeds" => TensorRef::from_array_view(token_embed.view())?,
                "position_ids" => TensorRef::from_array_view(step_pos.view())?,
                "past_keys" => TensorRef::from_array_view(keys.view())?,
                "past_values" => TensorRef::from_array_view(values.view())?,
            ];
            let step_outputs = self.decoder_step.run(step_inputs)?;

            let step_logits = step_outputs
                .get("logits")
                .ok_or_else(|| Qwen3Error::Decode("Missing 'logits' from decoder_step".into()))?
                .try_extract_array::<f32>()?;

            let next_token = argmax_slice(&step_logits, 0, vocab_size);
            output_tokens.push(next_token);
            pos += 1;

            if self
                .config
                .special_tokens
                .eos_token_ids
                .contains(&next_token)
            {
                break;
            }

            // Update KV cache
            keys = step_outputs
                .get("present_keys")
                .ok_or_else(|| Qwen3Error::Decode("Missing 'present_keys' from step".into()))?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dyn();
            values = step_outputs
                .get("present_values")
                .ok_or_else(|| Qwen3Error::Decode("Missing 'present_values' from step".into()))?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dyn();
        }

        Ok(self.tokenizer.decode(&output_tokens))
    }

    /// Full transcription pipeline: audio samples → text.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, Box<dyn std::error::Error>> {
        let mel = log_mel_spectrogram(samples);
        let mel_frames = mel.shape()[2];
        let audio_tokens = get_feat_extract_output_lengths(mel_frames);
        log::debug!(
            "Mel frames: {}, encoder output tokens: {}",
            mel_frames,
            audio_tokens
        );

        let audio_features = self.encode(&mel)?;
        log::debug!("Audio features shape: {:?}", audio_features.shape());

        self.greedy_decode(&audio_features, 512)
    }
}

/// Argmax over a 3D logits array at [0, pos, :].
fn argmax_slice(logits: &ndarray::ArrayViewD<'_, f32>, pos: usize, vocab_size: usize) -> i64 {
    let mut best_idx = 0i64;
    let mut best_val = f32::NEG_INFINITY;

    let shape = logits.shape();
    let stride_1 = shape[2]; // vocab_size
    let base = pos * stride_1;

    for v in 0..vocab_size {
        let val = logits
            .as_slice()
            .map_or_else(|| logits[[0, pos, v].as_ref()], |s| s[base + v]);
        if val > best_val {
            best_val = val;
            best_idx = v as i64;
        }
    }
    best_idx
}
