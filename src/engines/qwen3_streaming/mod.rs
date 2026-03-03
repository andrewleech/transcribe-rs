//! Qwen3-ASR streaming speech recognition engine.
//!
//! This module implements [`StreamingTranscriptionEngine`] for the Qwen3-ASR
//! ONNX model. Unlike natively streaming architectures (e.g. Nemotron), Qwen3-ASR
//! processes full audio at once — the streaming adapter buffers audio and periodically
//! re-runs the full encoder → decoder pipeline, emitting confirmed text incrementally.
//!
//! # Requirements
//!
//! The model directory must contain:
//! - `encoder.onnx` (+ `.data`) — audio encoder
//! - `decoder_init.onnx` (+ `.data`) — decoder prefill
//! - `decoder_step.onnx` (+ `.data`) — decoder autoregressive step
//! - `embed_tokens.bin` — embedding matrix (raw f32)
//! - `config.json` — model configuration
//! - `tokenizer.json` — HuggingFace BPE tokenizer
//!
//! # Example
//!
//! ```rust,no_run
//! use transcribe_rs::{StreamingTranscriptionEngine, engines::qwen3_streaming::Qwen3StreamingEngine};
//! use std::path::PathBuf;
//!
//! let mut engine = Qwen3StreamingEngine::new();
//! engine.load_model(&PathBuf::from("models/qwen3-asr-0.6b"))?;
//!
//! // Push 16kHz mono f32 audio in chunks
//! # let audio_chunks: Vec<Vec<f32>> = vec![];
//! for chunk in &audio_chunks {
//!     let segments = engine.push_samples(chunk)?;
//!     for seg in &segments {
//!         print!("{}", seg.text);
//!         if seg.is_endpoint {
//!             println!();
//!         }
//!     }
//! }
//! println!("\nFinal: {}", engine.get_transcript());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Re-export shared modules from the base qwen3 engine.
pub use crate::engines::qwen3::{config, mel, prompt};

mod engine;

pub use engine::{Qwen3StreamingEngine, Qwen3StreamingModelParams};
