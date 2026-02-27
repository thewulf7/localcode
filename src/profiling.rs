use anyhow::Result;
use serde::Deserialize;

pub struct HardwareProfile {
    pub vram_gb: f32,
    pub ram_gb: f32,
    pub compute_capability: ComputeCapability,
    pub recommended_models: Vec<RecommendedModel>,
}

#[derive(Debug, Clone)]
pub struct RecommendedModel {
    pub name: String,
    pub score: f32,
    pub best_quant: String,
}

#[derive(Deserialize)]
struct LlmfitOutput {
    models: Vec<LlmfitModel>,
    system: LlmfitSystem,
}

#[derive(Deserialize)]
struct LlmfitModel {
    name: String,
    score: f32,
    best_quant: String,
}

#[derive(Deserialize)]
struct LlmfitSystem {
    total_ram_gb: f32,
    gpu_vram_gb: Option<f32>,
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

    let mut recommended_models = Vec::new();
    let mut ram_gb = 32.0;
    let mut vram_gb = 8.0;

    // Run llmfit recommend --json
    if let Ok(output) = tokio::process::Command::new("llmfit")
        .args(&["recommend", "--json"])
        .output()
        .await
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(parsed) = serde_json::from_str::<LlmfitOutput>(&stdout) {
                ram_gb = parsed.system.total_ram_gb;
                if let Some(vram) = parsed.system.gpu_vram_gb {
                    vram_gb = vram;
                }
                for model in parsed.models {
                    recommended_models.push(RecommendedModel {
                        name: model.name,
                        score: model.score,
                        best_quant: model.best_quant,
                    });
                }
            }
        } else {
            println!("⚠️ llmfit failed. Using fallback HW detection.");
        }
    }
    
    // If llmfit didn't execute properly, Fallback hardware detection here...
    if recommended_models.is_empty() {
        #[cfg(target_os = "windows")]
        {
            // Get System RAM
            if let Ok(output) = tokio::process::Command::new("wmic")
                .args(&["computersystem", "get", "TotalPhysicalMemory"])
                .output()
                .await 
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let trimmed = line.trim();
                    if trimmed.chars().all(char::is_numeric) && !trimmed.is_empty() {
                        if let Ok(bytes) = trimmed.parse::<u64>() {
                            ram_gb = (bytes as f64 / 1_073_741_824.0) as f32;
                        }
                    }
                }
            }

            // Get VRAM
            if let Ok(output) = tokio::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                .output()
                .await 
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let mut total_vram_mb: u64 = 0;
                for line in stdout.lines() {
                    let trimmed = line.trim();
                    if trimmed.chars().all(char::is_numeric) && !trimmed.is_empty() {
                        if let Ok(mb) = trimmed.parse::<u64>() {
                            total_vram_mb += mb;
                        }
                    }
                }
                if total_vram_mb > 0 {
                    vram_gb = (total_vram_mb as f64 / 1024.0) as f32;
                }
            }
        }
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

    #[test]
    fn test_parse_llmfit_json() {
        let json = r#"{
            "models": [
                {
                    "name": "llama3-8b-instruct",
                    "score": 0.95,
                    "best_quant": "Q4_K_M"
                }
            ],
            "system": {
                "total_ram_gb": 32.0,
                "gpu_vram_gb": 8.0
            }
        }"#;

        let parsed: LlmfitOutput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.system.total_ram_gb, 32.0);
        assert_eq!(parsed.system.gpu_vram_gb, Some(8.0));
        assert_eq!(parsed.models.len(), 1);
        assert_eq!(parsed.models[0].name, "llama3-8b-instruct");
        assert_eq!(parsed.models[0].score, 0.95);
        assert_eq!(parsed.models[0].best_quant, "Q4_K_M");
    }

    #[test]
    fn test_parse_llmfit_json_no_gpu() {
        let json = r#"{
            "models": [],
            "system": {
                "total_ram_gb": 16.0,
                "gpu_vram_gb": null
            }
        }"#;

        let parsed: LlmfitOutput = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.system.total_ram_gb, 16.0);
        assert_eq!(parsed.system.gpu_vram_gb, None);
        assert!(parsed.models.is_empty());
    }
}
