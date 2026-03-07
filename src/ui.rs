use crate::profiling::HardwareProfile;
use anyhow::Result;
use inquire::Confirm;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelSelection {
    pub name: String,
    pub quant: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LlamaServerArgs {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ctx_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub n_gpu_layers: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub flash_attn: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cache_type_k: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cache_type_v: Option<String>,
    #[serde(flatten, default)]
    pub extra_args: HashMap<String, serde_json::Value>,
}

impl LlamaServerArgs {
    pub fn from_hardware(profile: &HardwareProfile, models: &[ModelSelection]) -> Self {
        let has_gpu = profile.vram_gb >= 1.0;

        let ctx_size = if has_gpu {
            if profile.vram_gb >= 24.0 {
                65536
            } else if profile.vram_gb >= 14.0 {
                16384
            } else if profile.vram_gb >= 12.0 {
                8192
            } else if profile.vram_gb >= 8.0 {
                4096
            } else {
                2048
            }
        } else if profile.ram_gb >= 32.0 {
            8192
        } else if profile.ram_gb >= 16.0 {
            4096
        } else {
            2048
        };

        // Heuristic mapping for GPU layers based on VRAM
        let n_gpu_layers = if has_gpu {
            if profile.vram_gb >= 12.0 {
                999 // Offload entirely (fits most 8B)
            } else if profile.vram_gb >= 8.0 {
                33 // Typically enough for full 8B, keeps system stable
            } else if profile.vram_gb >= 6.0 {
                24
            } else if profile.vram_gb >= 4.0 {
                16
            } else if profile.vram_gb >= 2.0 {
                8
            } else {
                0
            }
        } else {
            0
        };

        // Heuristic for KV Cache Type based on primary model quant
        let mut kv_quant = "f16".to_string();
        if has_gpu {
            if let Some(m) = models.first() {
                if let Some(q) = &m.quant {
                    let q_lower = q.to_lowercase();
                    if q_lower.contains("8") {
                        kv_quant = "q8_0".to_string();
                    } else if q_lower.contains("4") || q_lower.contains("5") {
                        kv_quant = "q4_0".to_string();
                    } else {
                        kv_quant = "q4_0".to_string(); // Safest fallback for VRAM limits
                    }
                } else {
                    kv_quant = "q4_0".to_string();
                }
            } else {
                kv_quant = "q4_0".to_string();
            }
        }

        let mut extra_args = HashMap::new();
        extra_args.insert("slot-save-path".to_string(), serde_json::json!("/models"));

        LlamaServerArgs {
            ctx_size: Some(ctx_size),
            n_gpu_layers: Some(n_gpu_layers),
            flash_attn: Some(if has_gpu {
                "on".to_string()
            } else {
                "off".to_string()
            }),
            cache_type_k: Some(kv_quant.clone()),
            cache_type_v: Some(kv_quant),
            extra_args,
        }
    }

    pub fn to_cli_args(&self) -> String {
        let mut args = String::new();

        if let Some(v) = self.ctx_size {
            args.push_str(&format!(" --ctx-size {}", v));
        }
        if let Some(v) = self.n_gpu_layers {
            args.push_str(&format!(" --n-gpu-layers {}", v));
        }
        if let Some(v) = &self.flash_attn
            && !v.is_empty()
        {
            args.push_str(&format!(" --flash-attn {}", v));
        }
        if let Some(v) = &self.cache_type_k {
            args.push_str(&format!(" --cache-type-k {}", v));
        }
        if let Some(v) = &self.cache_type_v {
            args.push_str(&format!(" --cache-type-v {}", v));
        }

        for (key, value) in &self.extra_args {
            if let Some(b) = value.as_bool() {
                if b {
                    args.push_str(&format!(" --{}", key));
                }
            } else if let Some(s) = value.as_str() {
                let expanded_s = shellexpand::tilde(s).to_string();
                args.push_str(&format!(" --{} {}", key, expanded_s));
            } else {
                args.push_str(&format!(" --{} {}", key, value));
            }
        }

        args.trim().to_string()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct InitConfig {
    pub models: Vec<ModelSelection>,
    pub run_in_docker: bool,
    pub models_dir: String,
    pub port: u16,
    #[serde(default)]
    pub llama_server_args: Option<LlamaServerArgs>,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            run_in_docker: true,
            models_dir: "~/.opencode/models".to_string(),
            port: 8080,
            llama_server_args: None,
        }
    }
}

const AVAILABLE_MODELS: &[&str] = &[
    "llama3-8b-instruct",
    "mixtral-8x7b-instruct",
    "llama3-70b-instruct",
    "phi3-mini",
    "gemma-2b-it",
    "qwen2-7b-instruct",
    "mistral-7b-instruct",
];

pub fn prompt_user(
    args: &crate::InitArgs,
    profile: &HardwareProfile,
    recommended_model: &str,
) -> Result<(InitConfig, bool)> {
    let is_project_scoped = if args.yes {
        !args.global
    } else if args.global {
        false
    } else {
        let scope_choice = inquire::Select::new(
            "Where would you like to save this configuration?",
            vec![
                "Locally (Current project directory only)",
                "Globally (~/.config/localcode/)",
            ],
        )
        .prompt()?;
        scope_choice.starts_with("Locally")
    };

    if args.yes {
        let models = if let Some(ref m_list) = args.models {
            m_list
                .iter()
                .map(|name| ModelSelection {
                    name: name.clone(),
                    quant: None,
                })
                .collect()
        } else {
            // Prefer Coding models in auto-mode
            let coding_combo = profile.recommended_combos.iter().find(|c| {
                c.standard_model.category.contains("Code")
                    || c.autocomplete_model.category.contains("Code")
            });

            if let Some(combo) = coding_combo {
                vec![
                    ModelSelection {
                        name: combo.standard_model.name.clone(),
                        quant: Some(combo.standard_model.best_quant.clone()),
                    },
                    ModelSelection {
                        name: combo.autocomplete_model.name.clone(),
                        quant: Some(combo.autocomplete_model.best_quant.clone()),
                    },
                ]
            } else if let Some(model) = profile
                .recommended_models
                .iter()
                .find(|m| m.category.contains("Code"))
            {
                vec![ModelSelection {
                    name: model.name.clone(),
                    quant: Some(model.best_quant.clone()),
                }]
            } else if !profile.recommended_combos.is_empty() {
                let combo = profile.recommended_combos.first().unwrap();
                vec![
                    ModelSelection {
                        name: combo.standard_model.name.clone(),
                        quant: Some(combo.standard_model.best_quant.clone()),
                    },
                    ModelSelection {
                        name: combo.autocomplete_model.name.clone(),
                        quant: Some(combo.autocomplete_model.best_quant.clone()),
                    },
                ]
            } else {
                vec![ModelSelection {
                    name: recommended_model.to_string(),
                    quant: profile
                        .recommended_models
                        .first()
                        .map(|m| m.best_quant.clone()),
                }]
            }
        };

        let llama_args = LlamaServerArgs::from_hardware(profile, &models);
        return Ok((
            InitConfig {
                models,
                run_in_docker: !args.no_docker,
                models_dir: args
                    .models_dir
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "~/.opencode/models".to_string()),
                port: args.port,
                llama_server_args: Some(llama_args),
            },
            is_project_scoped,
        ));
    }

    let default_choice = args
        .models
        .as_ref()
        .and_then(|m| m.first())
        .map(|s| s.as_str())
        .unwrap_or(recommended_model);

    let is_dynamic = !profile.recommended_models.is_empty();
    let has_combos = !profile.recommended_combos.is_empty();

    let use_combos = if has_combos && !args.yes {
        let choice = inquire::Select::new(
            "How would you like to run the models?",
            vec![
                "Single Model (Default)",
                "Normal Model + Small Autocomplete Model",
            ],
        )
        .prompt()?;
        choice.starts_with("Normal Model")
    } else {
        false
    };

    let all_options: Vec<String> = if use_combos {
        profile
            .recommended_combos
            .iter()
            .filter(|c| {
                (c.standard_model.category.contains("Code")
                    || c.autocomplete_model.category.contains("Code"))
                    && c.standard_model.name.to_lowercase().contains("instruct")
            })
            .map(|c| format!("{} (Score: {:.1})", c.name, c.score))
            .collect()
    } else if is_dynamic {
        profile
            .recommended_models
            .iter()
            .filter(|m| m.category.contains("Code") && m.name.to_lowercase().contains("instruct"))
            .map(|m| {
                format!(
                    "{} (Score: {:.1}, Quant: {})",
                    m.name, m.score, m.best_quant
                )
            })
            .collect()
    } else {
        AVAILABLE_MODELS
            .iter()
            .filter(|m| m.to_lowercase().contains("instruct") || m.to_lowercase().contains("-it"))
            .map(|&s| s.to_string())
            .collect()
    };

    let mut default_indices = Vec::new();

    // Add recommended model by default if it's in the list
    if let Some(idx) = all_options.iter().position(|x| x.contains(default_choice)) {
        default_indices.push(idx);
    }

    if default_indices.is_empty() && !all_options.is_empty() {
        default_indices.push(0);
    }

    default_indices.sort();
    default_indices.dedup();

    let selected_options = inquire::MultiSelect::new(
        "Which models would you like to install and use?",
        all_options,
    )
    .with_default(&default_indices)
    .with_help_message("Use Space to select/deselect, Enter to confirm. Type to filter.")
    .with_page_size(10)
    .prompt()?;

    if selected_options.is_empty() {
        anyhow::bail!("You must select at least one option.");
    }

    let mut selected_models = Vec::new();
    for opt in selected_options {
        let mut final_name = opt.clone();
        if let Some(idx) = opt.find(" (") {
            final_name = opt[..idx].to_string();
        }

        if use_combos {
            if let Some(combo) = profile
                .recommended_combos
                .iter()
                .find(|c| c.name == final_name)
            {
                selected_models.push(ModelSelection {
                    name: combo.standard_model.name.clone(),
                    quant: Some(combo.standard_model.best_quant.clone()),
                });
                selected_models.push(ModelSelection {
                    name: combo.autocomplete_model.name.clone(),
                    quant: Some(combo.autocomplete_model.best_quant.clone()),
                });
            }
        } else if is_dynamic {
            if let Some(model) = profile
                .recommended_models
                .iter()
                .find(|m| m.name == final_name)
            {
                selected_models.push(ModelSelection {
                    name: model.name.clone(),
                    quant: Some(model.best_quant.clone()),
                });
            }
        } else {
            selected_models.push(ModelSelection {
                name: final_name,
                quant: None,
            });
        }
    }

    let run_in_docker = Confirm::new("Do you want to run this using llama.cpp via Docker?")
        .with_default(!args.no_docker)
        .with_help_message("This will automatically download and start the model without installing extra dependencies natively.")
        .prompt()?;

    let default_models_dir = args
        .models_dir
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "~/.opencode/models".to_string());

    let models_dir_str = inquire::Text::new("Where would you like to save models locally?")
        .with_default(&default_models_dir)
        .prompt()?;

    Ok((
        InitConfig {
            models: selected_models.clone(),
            run_in_docker,
            models_dir: models_dir_str,
            port: args.port,
            llama_server_args: Some(LlamaServerArgs::from_hardware(profile, &selected_models)),
        },
        is_project_scoped,
    ))
}

pub fn display_config_instructions(config: &InitConfig) {
    let provider_url = format!("http://localhost:{}/v1", config.port);
    let standard_model = config
        .models
        .iter()
        .find(|m| !crate::runner::is_autocomplete_model(&m.name))
        .or_else(|| config.models.first())
        .map(|m| m.name.clone())
        .unwrap_or_else(|| "default".to_string());

    println!(
        "\n{}",
        crate::style("⚙️ Configuration Instructions").bold().cyan()
    );

    println!("\n{}", crate::style("--- OpenCode ---").bold().yellow());
    println!("To use your local server in OpenCode, update your `opencode.json`:");

    println!("{{");
    println!("  \"$schema\": \"https://opencode.ai/config.json\",");
    println!("  \"compaction\": {{");
    println!("    \"auto\": true,");
    println!("    \"prune\": true,");
    println!("    \"reserved\": 3000");
    println!("  }},");
    println!("  \"provider\": {{");
    println!("    \"localcode\": {{");
    println!("      \"models\": {{");
    println!("        \"{}\": {{", standard_model);
    println!("          \"name\": \"{}\",", standard_model);
    println!("        }}");
    println!("      }},");
    println!("      \"name\": \"LocalCode\",");
    println!("      \"npm\": \"@ai-sdk/openai-compatible\",");
    println!("      \"options\": {{");
    println!("        \"provider\": \"openai\",");
    println!("        \"baseURL\": \"{}\"", provider_url);
    println!("      }}");
    println!("    }}");
    println!("  }}");
    println!("}}");

    println!("\n{}", crate::style("--- Claude Code ---").bold().yellow());
    println!("To use your local server with Claude Code, run these commands in your terminal:");
    let shell_cmd = if cfg!(target_os = "windows") {
        "set"
    } else {
        "export"
    };
    println!("{} ANTHROPIC_BASE_URL=\"{}\"", shell_cmd, provider_url);
    println!("{} ANTHROPIC_API_KEY=\"sk-localcode\"", shell_cmd);
    println!("claude");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_selection_serialize() {
        let selection = ModelSelection {
            name: "test-model".to_string(),
            quant: Some("Q4".to_string()),
        };
        let serialized = serde_json::to_string(&selection).unwrap();
        assert!(serialized.contains("test-model"));
        assert!(serialized.contains("Q4"));
    }

    #[test]
    fn test_model_selection_deserialize() {
        let json = r#"{"name":"phi3-mini","quant":null}"#;
        let selection: ModelSelection = serde_json::from_str(json).unwrap();
        assert_eq!(selection.name, "phi3-mini");
        assert_eq!(selection.quant, None);
    }

    #[test]
    fn test_init_config_serialize() {
        let config = InitConfig {
            models: vec![ModelSelection {
                name: "test".to_string(),
                quant: None,
            }],
            run_in_docker: true,
            models_dir: "/tmp/models".to_string(),
            port: 8080,
            llama_server_args: None,
        };
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(serialized.contains("run_in_docker"));
        assert!(serialized.contains(r#""port":8080"#));
    }

    #[test]
    fn test_llama_server_args_from_hardware_gpu() {
        let profile = HardwareProfile {
            vram_gb: 16.0,
            ram_gb: 32.0,
            compute_capability: crate::profiling::ComputeCapability::High,
            recommended_models: vec![],
            recommended_combos: vec![],
        };
        let models = vec![ModelSelection {
            name: "test-model".to_string(),
            quant: Some("Q8_0".to_string()),
        }];
        let args = LlamaServerArgs::from_hardware(&profile, &models);
        assert_eq!(args.ctx_size, Some(16384));
        assert_eq!(args.n_gpu_layers, Some(999));
        assert_eq!(args.flash_attn, Some("on".to_string()));
        assert_eq!(args.cache_type_k, Some("q8_0".to_string()));
        assert_eq!(
            args.extra_args.get("slot-save-path").unwrap(),
            &serde_json::json!("/models")
        );
    }

    #[test]
    fn test_llama_server_args_to_cli_gpu() {
        let mut extra_args = HashMap::new();
        extra_args.insert("numa".to_string(), serde_json::json!("numactl"));
        extra_args.insert("mlock".to_string(), serde_json::json!(true));
        let args = LlamaServerArgs {
            ctx_size: Some(4096),
            n_gpu_layers: Some(999),
            flash_attn: Some("auto".to_string()),
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("q8_0".to_string()),
            extra_args,
        };
        let cli = args.to_cli_args();
        assert!(cli.contains("--ctx-size 4096"));
        assert!(cli.contains("--n-gpu-layers 999"));
        assert!(cli.contains("--flash-attn auto"));
        assert!(cli.contains("--cache-type-k q8_0"));
        assert!(cli.contains("--numa numactl"));
        assert!(cli.contains("--mlock"));
    }

    #[test]
    fn test_llama_server_args_to_cli_cpu() {
        let args = LlamaServerArgs {
            ctx_size: Some(2048),
            n_gpu_layers: Some(0),
            flash_attn: Some("off".to_string()),
            cache_type_k: Some("f16".to_string()),
            cache_type_v: Some("f16".to_string()),
            extra_args: HashMap::new(),
        };
        let cli = args.to_cli_args();
        assert!(cli.contains("--ctx-size 2048"));
        assert!(cli.contains("--n-gpu-layers 0"));
        assert!(cli.contains("--flash-attn off"));
        assert!(cli.contains("--cache-type-k f16"));
    }
    #[test]
    fn test_llama_server_args_deserialize_dynamic() {
        let json_payload = r#"{
            "ctx_size": 4096,
            "n_gpu_layers": 999,
            "flash_attn": "on",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "numa": "numactl",
            "threads": 8,
            "mlock": true,
            "no_mmap": false
        }"#;

        let args: LlamaServerArgs =
            serde_json::from_str(json_payload).expect("Failed to deserialize");
        let cli = args.to_cli_args();

        assert!(cli.contains("--ctx-size 4096"));
        assert!(cli.contains("--n-gpu-layers 999"));
        assert!(cli.contains("--flash-attn on"));
        assert!(cli.contains("--cache-type-k q8_0"));
        assert!(cli.contains("--numa numactl"));
        assert!(cli.contains("--threads 8"));
        assert!(cli.contains("--mlock"));
        assert!(!cli.contains("--no_mmap")); // False bools are ignored in my logic
    }

    #[test]
    fn test_llama_server_args_deserialize_optional() {
        let json_payload = r#"{
            "threads": 8
        }"#;

        let args: LlamaServerArgs =
            serde_json::from_str(json_payload).expect("Failed to deserialize");
        let cli = args.to_cli_args();
        assert!(cli.contains("--threads 8"));
        assert!(!cli.contains("--ctx-size"));
    }
}
