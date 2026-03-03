//! Batch (non-streaming) `TranscriptionEngine` implementation for Qwen3-ASR.

use std::path::{Path, PathBuf};

use crate::TranscriptionEngine;

use super::model::{Qwen3AsrModel, Qwen3ModelOptions};

/// Parameters for loading a Qwen3-ASR model.
#[derive(Debug, Clone, Default)]
pub struct Qwen3ModelParams {
    /// Use quantized (int8) model files if available.
    pub quantized: bool,
}

/// Parameters for configuring Qwen3-ASR inference.
#[derive(Debug, Clone, Default)]
pub struct Qwen3InferenceParams {
    /// Maximum number of tokens to decode. Default: 512.
    pub max_tokens: Option<usize>,
}

/// Errors produced by [`Qwen3Engine`].
#[derive(thiserror::Error, Debug)]
pub enum Qwen3EngineError {
    #[error("Model not loaded. Call load_model() first.")]
    ModelNotLoaded,
    #[error("Inference error: {0}")]
    Inference(String),
}

/// Batch transcription engine backed by Qwen3-ASR.
///
/// Processes complete audio in a single encoder-decoder pass, suitable for
/// the standard record-then-transcribe workflow.
pub struct Qwen3Engine {
    model: Option<Qwen3AsrModel>,
    loaded_model_path: Option<PathBuf>,
}

impl Qwen3Engine {
    pub fn new() -> Self {
        Self {
            model: None,
            loaded_model_path: None,
        }
    }
}

impl Default for Qwen3Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Qwen3Engine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for Qwen3Engine {
    type InferenceParams = Qwen3InferenceParams;
    type ModelParams = Qwen3ModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.unload_model();

        let options = Qwen3ModelOptions {
            quantized: params.quantized,
        };
        let model = Qwen3AsrModel::new(model_path, &options)?;
        self.model = Some(model);
        self.loaded_model_path = Some(model_path.to_path_buf());

        log::info!("Loaded Qwen3-ASR model from {:?}", model_path);
        Ok(())
    }

    fn unload_model(&mut self) {
        if self.model.is_some() {
            log::debug!("Unloading Qwen3-ASR model");
            self.model = None;
            self.loaded_model_path = None;
        }
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        _params: Option<Self::InferenceParams>,
    ) -> Result<crate::TranscriptionResult, Box<dyn std::error::Error>> {
        let model = self
            .model
            .as_mut()
            .ok_or(Qwen3EngineError::ModelNotLoaded)?;

        let text = model
            .transcribe(&samples)
            .map_err(|e| Qwen3EngineError::Inference(format!("{e}")))?;

        Ok(crate::TranscriptionResult {
            text,
            segments: None,
        })
    }
}
