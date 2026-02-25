use std::path::PathBuf;
use transcribe_rs::audio::read_wav_samples;
use transcribe_rs::engines::qwen3_streaming::{Qwen3StreamingEngine, Qwen3StreamingModelParams};
use transcribe_rs::StreamingTranscriptionEngine;

const MODEL_PATH: &str = "models/qwen3-asr-0.6b";

fn get_model_and_audio() -> Option<(PathBuf, PathBuf)> {
    let model_path = PathBuf::from(MODEL_PATH);
    let audio_path = PathBuf::from("samples/jfk.wav");

    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path);
        return None;
    }
    if !audio_path.exists() {
        eprintln!("Skipping test: audio not found at {:?}", audio_path);
        return None;
    }
    Some((model_path, audio_path))
}

#[test]
fn test_jfk_streaming_transcription() {
    let (model_path, audio_path) = match get_model_and_audio() {
        Some(v) => v,
        None => return,
    };

    let mut engine = Qwen3StreamingEngine::new();
    engine
        .load_model(&model_path)
        .expect("Failed to load Qwen3-ASR model");

    let samples = read_wav_samples(&audio_path).expect("Failed to read audio");

    // Push all audio at once (will trigger decode after min_new_samples threshold)
    let segments = engine
        .push_samples(&samples)
        .expect("push_samples should not fail");

    // May or may not have emitted segments depending on audio length vs threshold
    let transcript = engine.get_transcript();
    let lower = transcript.to_lowercase();

    assert!(
        !transcript.is_empty(),
        "Streaming transcription should produce non-empty output"
    );
    assert!(
        lower.contains("my fellow") || lower.contains("country"),
        "Transcript should contain expected JFK content, got: {}",
        transcript
    );

    // Collect any segment text
    let seg_text: String = segments.iter().map(|s| s.text.clone()).collect();
    assert!(
        seg_text.is_empty() || transcript.contains(seg_text.trim()),
        "Emitted segments should be a prefix of the full transcript"
    );

    engine.unload_model();
}

#[test]
fn test_reset_clears_state() {
    let (model_path, audio_path) = match get_model_and_audio() {
        Some(v) => v,
        None => return,
    };

    let mut engine = Qwen3StreamingEngine::new();
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    let samples = read_wav_samples(&audio_path).expect("Failed to read audio");

    // Push all audio
    let _ = engine.push_samples(&samples);

    let before_reset = engine.get_transcript();
    assert!(
        !before_reset.is_empty(),
        "Should have transcript before reset"
    );

    engine.reset();
    let after_reset = engine.get_transcript();
    assert!(
        after_reset.is_empty(),
        "reset() should clear accumulated transcript"
    );
}

#[test]
fn test_new_engine_empty_transcript() {
    let engine = Qwen3StreamingEngine::new();
    assert!(
        engine.get_transcript().is_empty(),
        "Fresh engine should return empty transcript"
    );
}

#[test]
fn test_push_samples_without_model() {
    let mut engine = Qwen3StreamingEngine::new();
    let result = engine.push_samples(&[0.0; 8960]);
    assert!(
        result.is_err(),
        "push_samples without a loaded model should return an error"
    );
}

#[test]
fn test_incremental_push() {
    let (model_path, audio_path) = match get_model_and_audio() {
        Some(v) => v,
        None => return,
    };

    let mut engine = Qwen3StreamingEngine::new();
    // Use a shorter interval for test (2 seconds)
    let params = Qwen3StreamingModelParams {
        min_new_samples: Some(32000),
        ..Default::default()
    };
    engine
        .load_model_with_params(&model_path, params)
        .expect("Failed to load model");

    let samples = read_wav_samples(&audio_path).expect("Failed to read audio");

    // Push in 500ms chunks
    let chunk_size = 8000;
    let mut all_segments = Vec::new();
    for chunk in samples.chunks(chunk_size) {
        let segments = engine.push_samples(chunk).expect("push_samples failed");
        all_segments.extend(segments);
    }

    let transcript = engine.get_transcript();
    assert!(
        !transcript.is_empty(),
        "Incremental push should produce non-empty transcript"
    );
}
