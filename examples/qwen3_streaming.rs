use std::env;
use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::engines::qwen3_streaming::{Qwen3StreamingEngine, Qwen3StreamingModelParams};
use transcribe_rs::StreamingTranscriptionEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model_dir> <audio.wav>", args[0]);
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let wav_path = PathBuf::from(&args[2]);

    // Read audio
    let reader = hound::WavReader::open(&wav_path)?;
    let spec = reader.spec();
    let audio_duration = reader.duration() as f64 / spec.sample_rate as f64;
    println!("Audio: {:.2}s", audio_duration);

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;

    // Load model
    let mut engine = Qwen3StreamingEngine::new();
    let params = Qwen3StreamingModelParams::default();

    println!("Loading Qwen3-ASR model from {:?}", model_dir);
    let load_start = Instant::now();
    engine.load_model_with_params(&model_dir, params)?;
    println!("Model loaded in {:.2?}", load_start.elapsed());

    // Simulate streaming: push 100ms chunks
    let chunk_size = 1600; // 100ms at 16kHz
    println!("Streaming {} samples in {}ms chunks...", samples.len(), 100);

    let transcribe_start = Instant::now();
    for chunk in samples.chunks(chunk_size) {
        let segments = engine.push_samples(chunk)?;
        for seg in &segments {
            print!("{}", seg.text);
            if seg.is_endpoint {
                println!(" [END]");
            }
        }
    }
    let transcribe_duration = transcribe_start.elapsed();
    println!();

    println!("Transcription completed in {:.2?}", transcribe_duration);
    let speedup = audio_duration / transcribe_duration.as_secs_f64();
    println!("Real-time factor: {:.2}x", speedup);
    println!("Full transcript: {}", engine.get_transcript());

    engine.unload_model();
    Ok(())
}
