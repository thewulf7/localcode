use anyhow::Result;

pub struct HardwareProfile {
    pub vram_gb: f32,
    pub ram_gb: f32,
    #[allow(dead_code)]
    pub compute_capability: ComputeCapability,
    pub recommended_models: Vec<RecommendedModel>,
}

#[derive(Debug, Clone)]
pub struct RecommendedModel {
    pub name: String,
    pub score: f32,
    pub best_quant: String,
}

#[allow(dead_code)]
pub enum ComputeCapability {
    Low,
    Medium,
    High,
    Ultra,
}

pub async fn profile_hardware() -> Result<HardwareProfile> {
    println!("🔍 Profiling hardware capabilities via llmfit...");

    let specs = llmfit_core::hardware::SystemSpecs::detect();
    let ram_gb = specs.total_ram_gb as f32;
    let vram_gb = specs.total_gpu_vram_gb.or(specs.gpu_vram_gb).unwrap_or(0.0) as f32;

    let db = llmfit_core::models::ModelDatabase::new();
    let models = db.models_fitting_system(
        specs.available_ram_gb,
        specs.has_gpu,
        specs.total_gpu_vram_gb.or(specs.gpu_vram_gb),
    );

    let mut fits = Vec::new();
    for m in models {
        fits.push(llmfit_core::fit::ModelFit::analyze(m, &specs));
    }

    let ranked = llmfit_core::fit::rank_models_by_fit(fits);

    let mut recommended_models = Vec::new();
    for fit in ranked {
        recommended_models.push(RecommendedModel {
            name: fit.model.name.clone(),
            score: fit.score as f32,
            best_quant: fit.best_quant.clone(),
        });
    }

    Ok(HardwareProfile {
        vram_gb,
        ram_gb,
        compute_capability: ComputeCapability::Medium,
        recommended_models,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_profile_hardware_returns_models() {
        // Run the hardware profiling, which uses llmfit_core underneath.
        let profile = profile_hardware().await.expect("Failed to profile hardware");

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
