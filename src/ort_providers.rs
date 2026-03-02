use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(feature = "webgpu")]
use ort::execution_providers::WebGPUExecutionProvider;

/// Runtime selection of which GPU execution provider to use.
///
/// Set once at startup (or when the user changes the setting) via
/// [`set_gpu_provider`].  Read by [`execution_providers`] each time an
/// ORT session is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GpuProvider {
    /// Try all compiled-in GPU EPs, then CPU (default).
    Auto = 0,
    /// CPU only — no GPU acceleration.
    CpuOnly = 1,
    /// DirectML (Windows).
    DirectMl = 2,
    /// CUDA (Linux).
    Cuda = 3,
    /// CoreML (macOS).
    CoreMl = 4,
    /// WebGPU (cross-platform).
    WebGpu = 5,
}

impl GpuProvider {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => GpuProvider::CpuOnly,
            2 => GpuProvider::DirectMl,
            3 => GpuProvider::Cuda,
            4 => GpuProvider::CoreMl,
            5 => GpuProvider::WebGpu,
            _ => GpuProvider::Auto,
        }
    }
}

static GPU_PROVIDER: AtomicU8 = AtomicU8::new(GpuProvider::Auto as u8);

/// Set the runtime GPU provider selection.
///
/// Call this before loading any models.  The value is read by
/// [`execution_providers`] when building ORT sessions.
pub fn set_gpu_provider(provider: GpuProvider) {
    GPU_PROVIDER.store(provider as u8, Ordering::Relaxed);
}

/// Read the current GPU provider selection.
pub fn get_gpu_provider() -> GpuProvider {
    GpuProvider::from_u8(GPU_PROVIDER.load(Ordering::Relaxed))
}

/// Return which GPU providers are available in this build.
///
/// Always includes `Auto` and `CpuOnly`; GPU-specific variants are
/// present only when the corresponding Cargo feature is compiled in.
pub fn available_providers() -> Vec<GpuProvider> {
    #[allow(unused_mut)]
    let mut v = vec![GpuProvider::Auto, GpuProvider::CpuOnly];
    #[cfg(feature = "directml")]
    v.push(GpuProvider::DirectMl);
    #[cfg(feature = "cuda")]
    v.push(GpuProvider::Cuda);
    #[cfg(feature = "coreml")]
    v.push(GpuProvider::CoreMl);
    #[cfg(feature = "webgpu")]
    v.push(GpuProvider::WebGpu);
    v
}

/// Return an ordered list of execution providers to try.
///
/// Respects the runtime [`GpuProvider`] selection.  When a specific
/// provider is selected, only that provider + CPU fallback are returned.
/// In `Auto` mode, all compiled-in GPU EPs are tried before CPU.
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    let selection = get_gpu_provider();

    match selection {
        GpuProvider::CpuOnly => {
            return vec![CPUExecutionProvider::default().build()];
        }
        #[cfg(feature = "directml")]
        GpuProvider::DirectMl => {
            return vec![
                DirectMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ];
        }
        #[cfg(feature = "cuda")]
        GpuProvider::Cuda => {
            return vec![
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ];
        }
        #[cfg(feature = "coreml")]
        GpuProvider::CoreMl => {
            return vec![
                CoreMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ];
        }
        #[cfg(feature = "webgpu")]
        GpuProvider::WebGpu => {
            return vec![
                WebGPUExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ];
        }
        GpuProvider::Auto => {}
        #[allow(unreachable_patterns)]
        _ => {
            // The selected provider wasn't compiled into this build.
            // Fall back to CPU-only rather than silently trying other GPU EPs.
            log::warn!(
                "Selected GPU provider {:?} is not available in this build, falling back to CPU",
                selection
            );
            return vec![CPUExecutionProvider::default().build()];
        }
    }

    // Auto: all compiled-in GPU EPs + CPU
    let mut providers = Vec::new();

    #[cfg(feature = "cuda")]
    providers.push(CUDAExecutionProvider::default().build());

    #[cfg(feature = "directml")]
    providers.push(DirectMLExecutionProvider::default().build());

    #[cfg(feature = "coreml")]
    providers.push(CoreMLExecutionProvider::default().build());

    #[cfg(feature = "webgpu")]
    providers.push(WebGPUExecutionProvider::default().build());

    providers.push(CPUExecutionProvider::default().build());
    providers
}

/// Return CPU-only execution providers.
///
/// Use this for models that contain operators with known GPU EP bugs
/// (e.g. Conv2d on DirectML producing NaN).  This bypasses the global
/// [`GpuProvider`] setting entirely.
pub fn cpu_execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![CPUExecutionProvider::default().build()]
}
