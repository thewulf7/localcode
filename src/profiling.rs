use anyhow::Result;
use llmfit_core::hardware::GpuBackend;

/// Upgrade a standard llmfit quant to the corresponding Unsloth Dynamic (UD-*_XL)
/// variant. UD quants use per-layer mixed precision: important attention layers get
/// higher precision while less sensitive layers stay compressed. The _XL suffix
/// means *more* layers promoted → better quality at ~12% size increase.
///
/// Unsloth publishes UD-GGUF files on HuggingFace under `unsloth/{model}-GGUF`.
fn upgrade_to_ud(quant: &str) -> String {
    match quant {
        "Q2_K"   => "UD-Q2_K_XL".to_string(),
        "Q3_K_M" => "UD-Q3_K_XL".to_string(),
        "Q4_K_M" => "UD-Q4_K_XL".to_string(),
        "Q5_K_M" => "UD-Q5_K_XL".to_string(),
        "Q6_K"   => "UD-Q6_K_XL".to_string(),
        "Q8_0"   => "UD-Q8_0".to_string(),
        other    => other.to_string(), // F16, mlx-*, etc. — unchanged
    }
}

pub struct HardwareProfile {
    pub vram_gb: f32,
    pub ram_gb: f32,
    pub cpu_cores: usize,
    #[allow(dead_code)]
    pub gpu_name: Option<String>,
    pub gpu_backend: GpuBackend,
    #[allow(dead_code)]
    pub gpu_count: u32,
    pub unified_memory: bool,
    pub recommended_models: Vec<RecommendedModel>,
    /// Total available memory for model loading (VRAM if GPU, RAM if CPU-only)
    pub available_memory_gb: f32,
}

#[derive(Debug, Clone)]
pub struct RecommendedModel {
    pub name: String,
    pub category: String,
    pub score: f32,
    pub best_quant: String,
    /// Estimated memory footprint in GB (weights + KV overhead)
    pub memory_gb: f32,
    /// Parameter count in billions (for VRAM budget calculations)
    #[allow(dead_code)]
    pub params_b: f32,
    /// Whether this is an autocomplete-sized model (≤3B)
    pub is_autocomplete: bool,
}

pub async fn profile_hardware() -> Result<HardwareProfile> {
    println!("🔍 Profiling hardware capabilities via llmfit...");

    let specs = llmfit_core::hardware::SystemSpecs::detect();
    let ram_gb = specs.total_ram_gb as f32;
    let vram_gb = specs.total_gpu_vram_gb.or(specs.gpu_vram_gb).unwrap_or(0.0) as f32;
    let cpu_cores = specs.total_cpu_cores;
    let gpu_name = specs.gpu_name.clone();
    let gpu_backend = specs.backend;
    let gpu_count = specs.gpu_count;
    let unified_memory = specs.unified_memory;

    let db = llmfit_core::models::ModelDatabase::new();
    let models = db.models_fitting_system(
        specs.available_ram_gb,
        specs.has_gpu,
        specs.total_gpu_vram_gb.or(specs.gpu_vram_gb),
    );

    let mut fits = Vec::new();
    for m in models {
        // Only keep GGUF-compatible models — llama-server cannot run MLX or GPTQ weights.
        let name_lower = m.name.to_lowercase();
        if name_lower.contains("mlx") || name_lower.contains("gptq") {
            continue;
        }
        fits.push(llmfit_core::fit::ModelFit::analyze(m, &specs));
    }

    let ranked = llmfit_core::fit::rank_models_by_fit(fits);

    let mut recommended_models = Vec::new();

    for fit in &ranked {
        let is_auto = crate::runner::is_autocomplete_model(&fit.model.name);
        recommended_models.push(RecommendedModel {
            name: fit.model.name.clone(),
            category: format!("{:?}", fit.model.use_case),
            score: fit.score as f32,
            best_quant: upgrade_to_ud(&fit.best_quant),
            memory_gb: fit.memory_required_gb as f32,
            params_b: fit.model.params_b() as f32,
            is_autocomplete: is_auto,
        });
    }

    let available_memory = if specs.has_gpu {
        specs
            .total_gpu_vram_gb
            .or(specs.gpu_vram_gb)
            .unwrap_or(specs.available_ram_gb)
    } else {
        specs.available_ram_gb
    };

    Ok(HardwareProfile {
        vram_gb,
        ram_gb,
        cpu_cores,
        gpu_name,
        gpu_backend,
        gpu_count,
        unified_memory,
        recommended_models,
        available_memory_gb: available_memory as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_profile_hardware_returns_models() {
        // Run the hardware profiling, which uses llmfit_core underneath.
        let profile = profile_hardware()
            .await
            .expect("Failed to profile hardware");

        // Asserts
        assert!(profile.vram_gb >= 0.0);
        assert!(profile.ram_gb >= 0.0);
        // It's possible on weak machines it returns empty, but mostly shouldn't.
        // We just ensure it runs and parses successfully.
        if !profile.recommended_models.is_empty() {
            let first = &profile.recommended_models[0];
            assert!(!first.name.is_empty());
            assert!(first.score > 0.0);
            assert!(!first.best_quant.is_empty());
        }
    }
}
