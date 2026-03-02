//! Diagnostic: compare CPU vs GPU EP outputs for Qwen3-ASR at each pipeline stage.
//!
//! Usage:
//!   cargo run --release --example diagnose_directml --features "qwen3-streaming,directml" <model_dir> <audio.wav>
//!   cargo run --release --example diagnose_directml --features "qwen3-streaming,webgpu" <model_dir> <audio.wav>
//!   cargo run --release --example diagnose_directml --features "qwen3-streaming,directml,webgpu" <model_dir> <audio.wav>

use std::env;
use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3, ArrayD};
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "webgpu")]
use ort::execution_providers::WebGPUExecutionProvider;

fn build_session(
    path: &Path,
    providers: Vec<ExecutionProviderDispatch>,
    opt_level: GraphOptimizationLevel,
) -> Session {
    Session::builder()
        .unwrap()
        .with_optimization_level(opt_level)
        .unwrap()
        .with_execution_providers(providers)
        .unwrap()
        .commit_from_file(path)
        .unwrap()
}

#[cfg(feature = "directml")]
fn build_session_no_dml_fusion(
    path: &Path,
    providers: Vec<ExecutionProviderDispatch>,
    opt_level: GraphOptimizationLevel,
) -> Session {
    Session::builder()
        .unwrap()
        .with_optimization_level(opt_level)
        .unwrap()
        .with_config_entry("ep.dml.disable_graph_fusion", "1")
        .unwrap()
        .with_execution_providers(providers)
        .unwrap()
        .commit_from_file(path)
        .unwrap()
}

fn compare_arrays(name: &str, cpu: &ArrayD<f32>, gpu: &ArrayD<f32>) {
    assert_eq!(cpu.shape(), gpu.shape(), "{name}: shape mismatch");

    let cpu_slice = cpu.as_slice().unwrap();
    let gpu_slice = gpu.as_slice().unwrap();

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut mismatch_count = 0usize;

    for (i, (&c, &g)) in cpu_slice.iter().zip(gpu_slice.iter()).enumerate() {
        if g.is_nan() {
            nan_count += 1;
            if nan_count <= 5 {
                println!("  {name}[{i}]: CPU={c:.6} GPU=NaN");
            }
            continue;
        }
        if g.is_infinite() {
            inf_count += 1;
            if inf_count <= 5 {
                println!("  {name}[{i}]: CPU={c:.6} GPU={g:.6}");
            }
            continue;
        }

        let abs_diff = (c - g).abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }

        let denom = c.abs().max(1e-8);
        let rel_diff = abs_diff / denom;
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }

        if abs_diff > 0.01 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                println!("  {name}[{i}]: CPU={c:.6} GPU={g:.6} (diff={abs_diff:.6})");
            }
        }
    }

    let total = cpu_slice.len();
    println!(
        "  {name}: shape={:?}, elements={total}, max_abs_diff={max_abs_diff:.6}, max_rel_diff={max_rel_diff:.6}, mismatches(>0.01)={mismatch_count}, NaN={nan_count}, Inf={inf_count}",
        cpu.shape()
    );

    if nan_count > 0 || mismatch_count > total / 10 {
        println!("  *** {name}: SIGNIFICANT DIVERGENCE ***");
    }

    let n = 10.min(total);
    print!("  {name} CPU first {n}: ");
    for v in &cpu_slice[..n] {
        print!("{v:.4} ");
    }
    println!();
    print!("  {name} GPU first {n}: ");
    for v in &gpu_slice[..n] {
        print!("{v:.4} ");
    }
    println!();
}

fn run_encoder(
    session: &mut Session,
    mel_dyn: &ArrayD<f32>,
) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    let inputs = ort::inputs!["mel" => TensorRef::from_array_view(mel_dyn.view())?];
    let outputs = session.run(inputs)?;
    Ok(outputs
        .get("audio_features")
        .unwrap()
        .try_extract_array::<f32>()?
        .to_owned()
        .into_dyn())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model_dir> <audio.wav>", args[0]);
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let wav_path = PathBuf::from(&args[2]);

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;
    println!(
        "Audio: {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f64 / 16000.0
    );

    let cpu_providers = vec![CPUExecutionProvider::default().build()];

    // --- Stage 1: Encoder ---
    let encoder_path = model_dir.join("encoder.onnx");

    let mel = transcribe_rs::engines::qwen3_streaming::mel::log_mel_spectrogram(
        &samples[..48000.min(samples.len())],
    );
    println!("Mel shape: {:?}", mel.shape());
    let mel_dyn = mel.clone().into_dyn();

    // CPU baseline
    let mut enc_cpu = build_session(
        &encoder_path,
        cpu_providers.clone(),
        GraphOptimizationLevel::Level3,
    );
    let cpu_enc_out = run_encoder(&mut enc_cpu, &mel_dyn)?;

    // --- DirectML tests ---
    #[cfg(feature = "directml")]
    {
        let dml_providers: Vec<ExecutionProviderDispatch> = vec![
            DirectMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ];

        let opt_levels = [
            ("Disable", GraphOptimizationLevel::Disable),
            ("Level1 (Basic)", GraphOptimizationLevel::Level1),
            ("Level2 (Extended)", GraphOptimizationLevel::Level2),
            ("Level3 (All)", GraphOptimizationLevel::Level3),
        ];

        for (name, level) in &opt_levels {
            println!("\n=== Encoder: DML opt_level={name} ===");
            let mut enc_dml = build_session(&encoder_path, dml_providers.clone(), *level);
            let dml_enc_out = run_encoder(&mut enc_dml, &mel_dyn)?;
            compare_arrays(&format!("encoder_{name}"), &cpu_enc_out, &dml_enc_out);
        }

        println!("\n=== Encoder: DML with ep.dml.disable_graph_fusion=1 ===");
        let mut enc_dml_no_fusion = build_session_no_dml_fusion(
            &encoder_path,
            dml_providers.clone(),
            GraphOptimizationLevel::Level3,
        );
        let dml_no_fusion_enc_out = run_encoder(&mut enc_dml_no_fusion, &mel_dyn)?;
        compare_arrays(
            "encoder_no_dml_fusion",
            &cpu_enc_out,
            &dml_no_fusion_enc_out,
        );

        // Decoder comparison with DML
        println!("\n=== Decoder Init: DML ===");
        run_decoder_comparison(
            &model_dir,
            &cpu_providers,
            &dml_providers,
            &cpu_enc_out,
            "DML",
        )?;
    }

    // --- WebGPU tests ---
    #[cfg(feature = "webgpu")]
    {
        let webgpu_providers: Vec<ExecutionProviderDispatch> = vec![
            WebGPUExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ];

        println!("\n=== Encoder: WebGPU EP ===");
        let mut enc_webgpu = build_session(
            &encoder_path,
            webgpu_providers.clone(),
            GraphOptimizationLevel::Level3,
        );
        let webgpu_enc_out = run_encoder(&mut enc_webgpu, &mel_dyn)?;
        compare_arrays("encoder_webgpu", &cpu_enc_out, &webgpu_enc_out);

        // Decoder comparison with WebGPU
        println!("\n=== Decoder Init: WebGPU ===");
        run_decoder_comparison(
            &model_dir,
            &cpu_providers,
            &webgpu_providers,
            &cpu_enc_out,
            "WebGPU",
        )?;
    }

    #[cfg(not(any(feature = "directml", feature = "webgpu")))]
    eprintln!("No GPU EP feature enabled. Use --features directml and/or --features webgpu.");

    println!("\nDone.");
    Ok(())
}

fn run_decoder_comparison(
    model_dir: &Path,
    cpu_providers: &[ExecutionProviderDispatch],
    gpu_providers: &[ExecutionProviderDispatch],
    cpu_enc_out: &ArrayD<f32>,
    ep_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let decoder_init_path = model_dir.join("decoder_init.onnx");

    let mut dec_init_cpu = build_session(
        &decoder_init_path,
        cpu_providers.to_vec(),
        GraphOptimizationLevel::Level3,
    );
    let mut dec_init_gpu = build_session(
        &decoder_init_path,
        gpu_providers.to_vec(),
        GraphOptimizationLevel::Level3,
    );

    let config =
        transcribe_rs::engines::qwen3_streaming::config::Qwen3AsrConfig::load(model_dir)?;
    let audio_token_count = cpu_enc_out.shape()[1];
    let prompt_ids =
        transcribe_rs::engines::qwen3_streaming::prompt::build_prompt_ids(
            &config.special_tokens,
            audio_token_count,
        );
    let seq_len = prompt_ids.len();
    println!("Prompt length: {seq_len}, audio tokens: {audio_token_count}");

    let embed_path = model_dir.join("embed_tokens.bin");
    let embed_data = std::fs::read(&embed_path)?;
    let [vocab_size, hidden_size] = config.embed_tokens_shape;
    let embed_floats: Vec<f32> = embed_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let embed_tokens = Array2::from_shape_vec((vocab_size, hidden_size), embed_floats)?;

    let mut embeds = Array2::<f32>::zeros((seq_len, hidden_size));
    for (i, &id) in prompt_ids.iter().enumerate() {
        let id = id as usize;
        if id < embed_tokens.nrows() {
            embeds.row_mut(i).assign(&embed_tokens.row(id));
        }
    }

    let (audio_start, audio_end) =
        transcribe_rs::engines::qwen3_streaming::prompt::get_audio_pad_range(
            &prompt_ids,
            config.special_tokens.audio_pad_token_id,
        )?;
    let audio_len = audio_end - audio_start;
    let enc_3d = Array3::from_shape_vec(
        (
            cpu_enc_out.shape()[0],
            cpu_enc_out.shape()[1],
            cpu_enc_out.shape()[2],
        ),
        cpu_enc_out.iter().cloned().collect(),
    )?;
    for i in 0..audio_len {
        embeds
            .row_mut(audio_start + i)
            .assign(&enc_3d.slice(ndarray::s![0, i, ..]));
    }

    let input_embeds = embeds
        .into_shape_with_order((1, seq_len, hidden_size))?
        .into_dyn();
    let position_ids =
        Array2::<i64>::from_shape_fn((1, seq_len), |(_, j)| j as i64).into_dyn();

    let (cpu_logits, cpu_keys, cpu_values) = {
        let inputs = ort::inputs![
            "input_embeds" => TensorRef::from_array_view(input_embeds.view())?,
            "position_ids" => TensorRef::from_array_view(position_ids.view())?,
        ];
        let out = dec_init_cpu.run(inputs)?;
        let logits = out
            .get("logits")
            .unwrap()
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dyn();
        let keys = out
            .get("present_keys")
            .unwrap()
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dyn();
        let values = out
            .get("present_values")
            .unwrap()
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dyn();
        (logits, keys, values)
    };

    let (gpu_logits, gpu_keys, gpu_values) = {
        let inputs = ort::inputs![
            "input_embeds" => TensorRef::from_array_view(input_embeds.view())?,
            "position_ids" => TensorRef::from_array_view(position_ids.view())?,
        ];
        let out = dec_init_gpu.run(inputs)?;
        let logits = out
            .get("logits")
            .unwrap()
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dyn();
        let keys = out
            .get("present_keys")
            .unwrap()
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dyn();
        let values = out
            .get("present_values")
            .unwrap()
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dyn();
        (logits, keys, values)
    };

    compare_arrays(
        &format!("dec_init_logits_{ep_name}"),
        &cpu_logits,
        &gpu_logits,
    );
    compare_arrays(
        &format!("dec_init_keys_{ep_name}"),
        &cpu_keys,
        &gpu_keys,
    );
    compare_arrays(
        &format!("dec_init_values_{ep_name}"),
        &cpu_values,
        &gpu_values,
    );

    let logits_shape = cpu_logits.shape();
    let last_pos = logits_shape[1] - 1;
    let vocab = logits_shape[2];
    let cpu_token = argmax(&cpu_logits, last_pos, vocab);
    let gpu_token = argmax(&gpu_logits, last_pos, vocab);
    println!(
        "  First token: CPU={cpu_token} {ep_name}={gpu_token} (match={})",
        cpu_token == gpu_token
    );

    Ok(())
}

fn argmax(logits: &ArrayD<f32>, pos: usize, vocab_size: usize) -> i64 {
    let slice = logits.as_slice().unwrap();
    let base = pos * vocab_size;
    let mut best_idx = 0i64;
    let mut best_val = f32::NEG_INFINITY;
    for v in 0..vocab_size {
        let val = slice[base + v];
        if val > best_val {
            best_val = val;
            best_idx = v as i64;
        }
    }
    best_idx
}
