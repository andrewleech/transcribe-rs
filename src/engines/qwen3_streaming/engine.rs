//! Streaming state machine that wraps `Qwen3AsrModel` to implement
//! `StreamingTranscriptionEngine`.
//!
//! Since Qwen3-ASR processes full audio at once, the streaming adapter:
//! 1. Buffers incoming audio samples.
//! 2. Every `min_new_samples` samples (default 3s), re-runs the full pipeline.
//! 3. Compares the new transcript with the previous one.
//! 4. Emits only text that appears in a stable prefix across two consecutive decodes.

use std::path::{Path, PathBuf};

use crate::{split_at_sentence_boundaries, StreamingSegment, StreamingTranscriptionEngine};

use super::model::{Qwen3AsrModel, Qwen3ModelOptions};

/// Default number of new samples before triggering a re-decode (3 seconds at 16kHz).
///
/// The confirm-before-emit strategy requires two consecutive decode passes to agree,
/// so first emitted text typically appears after ~2× this interval.
const DEFAULT_MIN_NEW_SAMPLES: usize = 48000;

/// Parameters for loading a Qwen3-ASR streaming model.
#[derive(Debug, Clone, Default)]
pub struct Qwen3StreamingModelParams {
    /// Use quantized (int8) model files if available.
    pub quantized: bool,
    /// Minimum new audio samples before re-running the decode pipeline.
    /// Default: 48000 (3 seconds at 16kHz).
    pub min_new_samples: Option<usize>,
}

/// Errors produced by [`Qwen3StreamingEngine`].
#[derive(thiserror::Error, Debug)]
pub enum Qwen3StreamingError {
    #[error("Model not loaded. Call load_model() first.")]
    ModelNotLoaded,
    #[error("Inference error: {0}")]
    Inference(String),
}

/// Streaming transcription engine backed by Qwen3-ASR-0.6B.
///
/// Audio is buffered and periodically re-run through the full encoder→decoder
/// pipeline. Text that appears consistently across consecutive decode passes
/// is emitted as confirmed output.
///
/// **Latency:** The confirm-before-emit strategy requires two consecutive decode
/// passes to produce a stable prefix, so the minimum latency to first emitted
/// text is approximately `2 × min_new_samples / sample_rate` (default: ~6 seconds).
pub struct Qwen3StreamingEngine {
    model: Option<Qwen3AsrModel>,
    loaded_model_path: Option<PathBuf>,
    state: StreamingState,
}

struct StreamingState {
    audio_buffer: Vec<f32>,
    /// Most recent full decode result (includes speculative text not yet emitted).
    latest_transcript: String,
    /// The previous decode result (for confirm-before-emit).
    previous_transcript: String,
    /// Text already returned via push_samples.
    emitted_text: String,
    /// Audio sample count at last decode.
    last_decode_len: usize,
    /// Re-run threshold in samples.
    min_new_samples: usize,
}

impl StreamingState {
    fn new(min_new_samples: usize) -> Self {
        Self {
            audio_buffer: Vec::new(),
            latest_transcript: String::new(),
            previous_transcript: String::new(),
            emitted_text: String::new(),
            last_decode_len: 0,
            min_new_samples,
        }
    }

    fn reset(&mut self) {
        self.audio_buffer.clear();
        self.latest_transcript.clear();
        self.previous_transcript.clear();
        self.emitted_text.clear();
        self.last_decode_len = 0;
    }
}

impl Qwen3StreamingEngine {
    /// Create a new engine with no model loaded.
    pub fn new() -> Self {
        Self {
            model: None,
            loaded_model_path: None,
            state: StreamingState::new(DEFAULT_MIN_NEW_SAMPLES),
        }
    }
}

impl Default for Qwen3StreamingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Qwen3StreamingEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

/// Find the longest common prefix between two strings, measured in bytes
/// but aligned to char boundaries.
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(ca, cb)| ca == cb)
        .map(|(c, _)| c.len_utf8())
        .sum()
}

impl StreamingTranscriptionEngine for Qwen3StreamingEngine {
    type ModelParams = Qwen3StreamingModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Qwen3StreamingModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.unload_model();

        let options = Qwen3ModelOptions {
            quantized: params.quantized,
        };
        let model = Qwen3AsrModel::new(model_path, &options)?;
        self.model = Some(model);
        self.loaded_model_path = Some(model_path.to_path_buf());
        self.state = StreamingState::new(params.min_new_samples.unwrap_or(DEFAULT_MIN_NEW_SAMPLES));

        log::info!("Loaded Qwen3-ASR streaming model from {:?}", model_path);
        Ok(())
    }

    fn unload_model(&mut self) {
        if self.model.is_some() {
            log::debug!("Unloading Qwen3-ASR streaming model");
            self.model = None;
            self.loaded_model_path = None;
            self.state.reset();
        }
    }

    fn push_samples(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<StreamingSegment>, Box<dyn std::error::Error>> {
        let model = self
            .model
            .as_mut()
            .ok_or(Qwen3StreamingError::ModelNotLoaded)?;

        self.state.audio_buffer.extend_from_slice(samples);

        // Check if enough new audio to warrant a re-decode
        let new_samples = self.state.audio_buffer.len() - self.state.last_decode_len;
        if new_samples < self.state.min_new_samples {
            return Ok(vec![]);
        }

        // Run full pipeline on accumulated buffer
        let new_transcript = model
            .transcribe(&self.state.audio_buffer)
            .map_err(|e| Qwen3StreamingError::Inference(format!("{e}")))?;

        self.state.last_decode_len = self.state.audio_buffer.len();

        // Confirm-before-emit: find stable prefix between previous and current decode
        let stable_len = common_prefix_len(&self.state.previous_transcript, &new_transcript);

        // Determine new text to emit (beyond what was already emitted)
        let emit_end = stable_len;
        let emit_start = self.state.emitted_text.len();
        let new_to_emit = if emit_end > emit_start {
            new_transcript[emit_start..emit_end].to_string()
        } else {
            String::new()
        };

        // Update state
        self.state.previous_transcript =
            std::mem::replace(&mut self.state.latest_transcript, new_transcript);

        if new_to_emit.is_empty() {
            return Ok(vec![]);
        }

        self.state.emitted_text.push_str(&new_to_emit);
        Ok(split_at_sentence_boundaries(&new_to_emit))
    }

    fn get_transcript(&self) -> String {
        self.state.latest_transcript.clone()
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_prefix_len() {
        assert_eq!(common_prefix_len("hello world", "hello there"), 6);
        assert_eq!(common_prefix_len("abc", "abc"), 3);
        assert_eq!(common_prefix_len("abc", "xyz"), 0);
        assert_eq!(common_prefix_len("", "abc"), 0);
        assert_eq!(common_prefix_len("abc", ""), 0);
    }

    #[test]
    fn test_new_engine_empty_transcript() {
        let engine = Qwen3StreamingEngine::new();
        assert!(engine.get_transcript().is_empty());
    }

    #[test]
    fn test_push_samples_without_model() {
        let mut engine = Qwen3StreamingEngine::new();
        let result = engine.push_samples(&[0.0; 16000]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut engine = Qwen3StreamingEngine::new();
        engine.state.latest_transcript = "hello".to_string();
        engine.state.emitted_text = "hello".to_string();
        engine.state.audio_buffer = vec![0.0; 1000];
        engine.reset();
        assert!(engine.get_transcript().is_empty());
        assert!(engine.state.audio_buffer.is_empty());
        assert!(engine.state.emitted_text.is_empty());
    }
}
