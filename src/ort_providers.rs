use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;

/// Return an ordered list of execution providers to try.
///
/// GPU providers (if compiled in) come first; CPU is always last as a
/// fallback.  ORT tries each provider in order and silently falls back to
/// the next if registration fails.
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    let mut providers = Vec::new();

    #[cfg(feature = "cuda")]
    providers.push(CUDAExecutionProvider::default().build());

    #[cfg(feature = "directml")]
    providers.push(DirectMLExecutionProvider::default().build());

    #[cfg(feature = "coreml")]
    providers.push(CoreMLExecutionProvider::default().build());

    providers.push(CPUExecutionProvider::default().build());
    providers
}

/// Return CPU-only execution providers.
///
/// Use this for models that contain operators with known GPU EP bugs
/// (e.g. Conv2d on DirectML producing NaN).
pub fn cpu_execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![CPUExecutionProvider::default().build()]
}
