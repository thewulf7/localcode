use crate::profiling::HardwareProfile;
use anyhow::Result;
use inquire::Confirm;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

/// Deserialize `flash_attn` tolerating old configs that stored a boolean.
/// `true` → `"on"`, `false` → `"off"`, string values passed through as-is.
fn deserialize_flash_attn<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let v: Option<serde_json::Value> = Option::deserialize(d)?;
    Ok(match v {
        None => None,
        Some(serde_json::Value::Bool(b)) => Some(if b { "on" } else { "off" }.to_string()),
        Some(serde_json::Value::String(s)) => Some(s),
        Some(other) => Some(other.to_string()),
    })
}

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
    #[serde(
        deserialize_with = "deserialize_flash_attn",
        skip_serializing_if = "Option::is_none",
        default
    )]
    pub flash_attn: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cache_type_k: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cache_type_v: Option<String>,
    #[serde(flatten, default)]
    pub extra_args: HashMap<String, serde_json::Value>,
}

impl LlamaServerArgs {
    /// Parse parameter count in billions from a model name (e.g. "Qwen2.5-Coder-7B" → 7.0).
    fn parse_params_b(name: &str) -> Option<f64> {
        let re = regex::Regex::new(r"(?i)[-_](\d+\.?\d*)[Bb](?:[-_]|$)").ok()?;
        re.captures(name)?.get(1)?.as_str().parse().ok()
    }

    /// Bytes-per-parameter for a quantization string, handling Unsloth Dynamic
    /// (UD-) and extended _XL / _XS variants that llmfit-core doesn't have.
    ///
    /// UD (Unsloth Dynamic) quants assign different precision per-layer based
    /// on sensitivity analysis, giving better quality at the same average bpp.
    /// The _XL suffix means more layers promoted to higher precision (~12% bigger).
    /// The _XS suffix trades precision for size (~10% smaller).
    fn normalize_quant_bpp(quant: &str) -> f64 {
        // Strip "UD-" prefix — the dynamic per-layer strategy doesn't change
        // the *average* bpp significantly vs. the same base quant.
        let stripped = quant.strip_prefix("UD-").unwrap_or(quant);

        if stripped.ends_with("_XL") {
            // _XL: more important layers quantized at higher precision.
            // Map Q4_K_XL → Q4_K_M's bpp × 1.12  (≈ 5.25 bpw for Q4 tier)
            let base = &stripped[..stripped.len() - 3]; // "Q4_K"
            let base_quant = format!("{}_M", base);
            llmfit_core::models::quant_bpp(&base_quant) * 1.12
        } else if stripped.ends_with("_XS") {
            // _XS: more aggressive quantization for smaller files.
            let base = &stripped[..stripped.len() - 3];
            let base_quant = format!("{}_M", base);
            llmfit_core::models::quant_bpp(&base_quant) * 0.90
        } else {
            // Standard quant or UD- with standard suffix (e.g. UD-Q8_0)
            llmfit_core::models::quant_bpp(stripped)
        }
    }

    /// KV cache memory multiplier relative to the f16 baseline used by llmfit.
    fn kv_cache_multiplier(cache_type: &str) -> f64 {
        match cache_type {
            "q4_0" | "q4_1" => 0.25,
            "q8_0" | "q8_1" => 0.5,
            "f16" => 1.0,
            "f32" => 2.0,
            _ => 1.0,
        }
    }

    /// Infer the native (training) context length from a model name.
    ///
    /// Most open-weights models embed context length in their GGUF metadata,
    /// but at `init` time we only have the HuggingFace name. This gives us a
    /// conservative default per model family.
    fn native_ctx_length(model_name: &str) -> u32 {
        let lower = model_name.to_lowercase();
        // Qwen 2.5 series: 32768 (7B/14B) or 131072 (32B/72B)
        if lower.contains("qwen2.5") || lower.contains("qwen-2.5") {
            if lower.contains("32b") || lower.contains("72b") {
                131072
            } else {
                32768
            }
        // Qwen 3 series: 40960 (all sizes)
        } else if lower.contains("qwen3") || lower.contains("qwen-3") {
            40960
        // Llama 3.1/3.2/3.3 series: 131072
        } else if lower.contains("llama-3.1") || lower.contains("llama-3.2") || lower.contains("llama-3.3") {
            131072
        // Llama 3 (original): 8192
        } else if lower.contains("llama-3") || lower.contains("llama3") {
            8192
        // DeepSeek V2/V3/R1: 131072
        } else if lower.contains("deepseek") {
            131072
        // Gemma 2: 8192
        } else if lower.contains("gemma-2") || lower.contains("gemma2") {
            8192
        // Phi-3/3.5: 131072 (long) or 4096 (mini default)
        } else if lower.contains("phi-3") || lower.contains("phi3") {
            if lower.contains("mini") {
                4096
            } else {
                131072
            }
        // Mistral / Codestral: 32768
        } else if lower.contains("mistral") || lower.contains("codestral") {
            32768
        // StarCoder2: 16384
        } else if lower.contains("starcoder") {
            16384
        // Conservative fallback
        } else {
            32768
        }
    }

    /// Calculate the maximum context size that fits in the given VRAM budget,
    /// capped by the model's native training context length.
    ///
    /// Uses the llmfit-core memory model:
    ///   `total = params_b × bpp + 0.000008 × params_b × ctx + 0.5`
    ///
    /// Rearranged for VRAM-limited max_ctx:
    ///   `max_ctx = (vram × 0.90 - params_b × bpp - 0.5) / (0.000008 × params_b × kv_mult)`
    ///
    /// Context is NEVER extended beyond native training length. YaRN rope
    /// scaling is disabled because it causes gibberish on small models
    /// during structured tool-call generation.
    pub fn calculate_max_ctx(
        vram_gb: f64,
        params_b: f64,
        model_quant: &str,
        kv_cache_type: &str,
        model_name: &str,
    ) -> u32 {
        let bpp = Self::normalize_quant_bpp(model_quant);
        let model_mem = params_b * bpp;
        let overhead = 0.5_f64; // CUDA/Metal context + compute buffers
        let usable_vram = vram_gb * 0.90; // 10% safety margin

        // Hard ceiling: never exceed the model's native training context.
        // Extending beyond this requires YaRN which causes gibberish on
        // small models during structured tool-call generation.
        let native_ctx = Self::native_ctx_length(model_name);

        let free_for_kv = usable_vram - model_mem - overhead;
        if free_for_kv <= 0.0 {
            // Not enough VRAM for KV cache — use minimum workable context.
            return native_ctx.min(8192);
        }

        let kv_mult = Self::kv_cache_multiplier(kv_cache_type);
        let kv_per_token_gb = 0.000008 * params_b * kv_mult;
        if kv_per_token_gb <= 0.0 {
            return native_ctx;
        }

        let vram_ctx = (free_for_kv / kv_per_token_gb) as u32;

        // Cap at native training context — no YaRN extension.
        let effective = vram_ctx.min(native_ctx);
        let rounded = (effective / 1024) * 1024;

        // Floor at 8192 to be minimally functional.
        rounded.max(8192)
    }

    pub fn from_hardware(profile: &HardwareProfile, models: &[ModelSelection]) -> Self {
        use llmfit_core::hardware::GpuBackend;

        let has_gpu = profile.vram_gb >= 1.0;
        let vram = profile.vram_gb as f64;

        // Extract primary model metadata
        let model_name = models.first().map(|m| m.name.as_str()).unwrap_or("");
        let params_b = models
            .first()
            .and_then(|m| Self::parse_params_b(&m.name))
            .unwrap_or(7.0);
        let model_quant = models
            .first()
            .and_then(|m| m.quant.as_deref())
            .unwrap_or("Q4_K_M");
        let bpp = Self::normalize_quant_bpp(model_quant);
        let model_mem = params_b * bpp;

        // Calculate VRAM reserved by secondary/autocomplete models (loaded simultaneously).
        // llama-swap keeps them persistent, so we must subtract their footprint from
        // the VRAM budget available for the primary model's KV cache / context.
        let secondary_vram: f64 = models
            .iter()
            .skip(1) // skip primary
            .map(|m| {
                let p = Self::parse_params_b(&m.name).unwrap_or(1.5);
                let q = m.quant.as_deref().unwrap_or("Q4_K_M");
                let b = Self::normalize_quant_bpp(q);
                // Model weights + ~0.3 GB overhead for its KV cache / compute buffers
                p * b + 0.3
            })
            .sum();
        let effective_vram = vram - secondary_vram;

        // ── KV cache quantization ──────────────────────────────────────────
        // Context stays within native training length (no YaRN), so q4_0
        // is safe. Prefer q8_0 when VRAM headroom allows for best quality.
        let kv_quant = if has_gpu {
            let headroom_after_model = effective_vram - model_mem - 0.5;
            if headroom_after_model > 2.0 {
                "q8_0".to_string()
            } else {
                "q4_0".to_string()
            }
        } else {
            "f16".to_string()
        };

        // ── Context size ───────────────────────────────────────────────────
        // Cap at native training context regardless of GPU/CPU.
        let native_ctx = Self::native_ctx_length(model_name);
        let ctx_size = if has_gpu {
            Self::calculate_max_ctx(effective_vram, params_b, model_quant, &kv_quant, model_name)
        } else {
            // CPU-only: use native context (RAM is plentiful but extending
            // beyond training length still causes gibberish).
            native_ctx
        };

        // ── GPU layer offload ──────────────────────────────────────────────
        let n_gpu_layers = if has_gpu {
            if profile.unified_memory {
                // Apple Silicon / unified memory: all layers always in shared pool
                999
            } else if effective_vram >= model_mem + 1.0 {
                999 // Full GPU offload — model fits with headroom
            } else {
                // Partial offload: estimate layers from available VRAM fraction.
                // Most transformer models have ~(params_b × 4) layers.
                let total_layers = (params_b * 4.0).round() as i32;
                let frac = (effective_vram / model_mem).min(1.0);
                ((total_layers as f64 * frac) as i32).max(0)
            }
        } else {
            0
        };

        // ── Flash attention ────────────────────────────────────────────────
        // Flash-attn is well-supported on CUDA and Metal. Vulkan/ROCm/SYCL
        // support varies; disable by default for safety.
        let flash_attn = if has_gpu {
            match profile.gpu_backend {
                GpuBackend::Cuda | GpuBackend::Metal => "on".to_string(),
                _ => "off".to_string(),
            }
        } else {
            "off".to_string()
        };

        // ── Threads ────────────────────────────────────────────────────────
        // For GPU inference the CPU is mostly idle (prompt preprocessing).
        // Use physical cores (assume 2 HW threads per core) minus 2 for OS.
        // For CPU inference, use most cores.
        let mut extra_args = HashMap::new();
        let physical_cores = (profile.cpu_cores / 2).max(1);
        let threads = if has_gpu {
            // GPU inference: CPU does tokenization + HTTP; don't starve the OS
            physical_cores.min(8).max(2)
        } else {
            // CPU inference: use most physical cores, leave 2 for OS
            (physical_cores - 1).max(2)
        };
        extra_args.insert("threads".to_string(), serde_json::json!(threads));

        // ── Parallel slots ─────────────────────────────────────────────────
        // Number of concurrent request slots. Each slot reserves ctx_size
        // tokens of KV cache. More slots = more VRAM.
        //   slot_kv_gb = 0.000008 × params_b × kv_mult × ctx_size
        //   max_slots = min(floor(free_vram / slot_kv_gb), 4)
        // Default to 1 slot for safety, up to 4 for large VRAM systems.
        let parallel = if has_gpu && effective_vram >= model_mem + 2.0 {
            let kv_mult = Self::kv_cache_multiplier(&kv_quant);
            let slot_kv_gb = 0.000008 * params_b * kv_mult * ctx_size as f64;
            if slot_kv_gb > 0.0 {
                let free_for_slots = (effective_vram * 0.85) - model_mem - 0.5;
                let max_slots = (free_for_slots / slot_kv_gb).floor() as u32;
                max_slots.clamp(1, 4)
            } else {
                1
            }
        } else {
            1
        };
        if parallel > 1 {
            extra_args.insert("parallel".to_string(), serde_json::json!(parallel));
        }

        // ── Memory lock ────────────────────────────────────────────────────
        // Prevent OS from swapping the model out of RAM/VRAM. Beneficial on
        // Linux (mmap); less relevant on macOS (already unified).
        #[cfg(not(target_os = "macos"))]
        {
            if has_gpu && vram >= model_mem + 1.0 {
                extra_args.insert("mlock".to_string(), serde_json::json!(true));
            }
        }

        extra_args.insert("slot-save-path".to_string(), serde_json::json!("/models"));

        LlamaServerArgs {
            ctx_size: Some(ctx_size),
            n_gpu_layers: Some(n_gpu_layers),
            flash_attn: Some(flash_attn),
            cache_type_k: Some(kv_quant.clone()),
            cache_type_v: Some(kv_quant),
            extra_args,
        }
    }

    /// Build lighter CLI args for a secondary/autocomplete model.
    ///
    /// The secondary model serves subagent requests (haiku aliases) which carry
    /// much smaller contexts than the primary conversation.  Its `--ctx-size`
    /// is capped to the model's native training context, and only 1 parallel
    /// slot is used to conserve VRAM.
    pub fn for_secondary_model(
        primary_args: &LlamaServerArgs,
        model: &ModelSelection,
        profile: &HardwareProfile,
    ) -> LlamaServerArgs {
        let model_name = &model.name;
        let params_b = Self::parse_params_b(model_name).unwrap_or(1.5);
        let native_ctx = Self::native_ctx_length(model_name);

        // Cap context to native training length, but at least 8192
        // for the subagent system prompt + tool defs.
        let ctx_size = native_ctx.max(8192);

        let has_gpu = profile.vram_gb >= 1.0;
        let model_quant = model.quant.as_deref().unwrap_or("Q4_K_M");

        // n_gpu_layers: same strategy as primary — if it fits, go full offload
        let bpp = Self::normalize_quant_bpp(model_quant);
        let model_mem = params_b * bpp;
        let n_gpu_layers = if has_gpu && (model_mem + 0.5) <= profile.vram_gb as f64 {
            999
        } else {
            0
        };

        // KV cache: q4_0 for secondary (conserve VRAM, no YaRN needed)
        let kv_quant = if has_gpu { "q4_0" } else { "f16" };

        let mut extra_args = std::collections::HashMap::new();

        // Inherit thread count from primary
        if let Some(threads) = primary_args.extra_args.get("threads") {
            extra_args.insert("threads".to_string(), threads.clone());
        }
        // Smaller batch sizes for the lighter model
        extra_args.insert("batch-size".to_string(), serde_json::json!(2048));
        extra_args.insert("ubatch-size".to_string(), serde_json::json!(512));
        // Greedy decoding for tool calls with strong anti-repetition
        extra_args.insert("temp".to_string(), serde_json::json!(0.0));
        extra_args.insert("top-k".to_string(), serde_json::json!(0));
        extra_args.insert("top-p".to_string(), serde_json::json!(1.0));
        extra_args.insert("repeat_penalty".to_string(), serde_json::json!(1.3));

        LlamaServerArgs {
            ctx_size: Some(ctx_size),
            n_gpu_layers: Some(n_gpu_layers),
            flash_attn: primary_args.flash_attn.clone(),
            cache_type_k: Some(kv_quant.to_string()),
            cache_type_v: Some(kv_quant.to_string()),
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
            // Auto-mode: pick best coding primary, then best autocomplete that fits
            let primary = profile
                .recommended_models
                .iter()
                .filter(|m| {
                    !m.is_autocomplete
                        && m.category.contains("Code")
                        && m.name.to_lowercase().contains("instruct")
                })
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(p) = primary {
                let mut models = vec![ModelSelection {
                    name: p.name.clone(),
                    quant: Some(p.best_quant.clone()),
                }];

                let remaining = profile.available_memory_gb - p.memory_gb;
                if let Some(sec) = profile
                    .recommended_models
                    .iter()
                    .filter(|m| m.is_autocomplete && m.memory_gb <= remaining && m.name != p.name)
                    .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
                {
                    models.push(ModelSelection {
                        name: sec.name.clone(),
                        quant: Some(sec.best_quant.clone()),
                    });
                }
                models
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

    // ── Step 1: Select the primary model ──────────────────────────────
    let is_dynamic = !profile.recommended_models.is_empty();

    // Build primary model options — non-autocomplete coding/instruct models
    let primary_options: Vec<String> = if is_dynamic {
        profile
            .recommended_models
            .iter()
            .filter(|m| {
                !m.is_autocomplete
                    && (m.category.contains("Code") || m.category.contains("Reasoning"))
                    && m.name.to_lowercase().contains("instruct")
            })
            .map(|m| {
                format!(
                    "{} ({}, {:.1} GB, Score: {:.1})",
                    m.name, m.best_quant, m.memory_gb, m.score
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

    if primary_options.is_empty() {
        anyhow::bail!("No compatible models found for your hardware.");
    }

    let default_choice = args
        .models
        .as_ref()
        .and_then(|m| m.first())
        .map(|s| s.as_str())
        .unwrap_or(recommended_model);

    let default_idx = primary_options
        .iter()
        .position(|x| x.contains(default_choice))
        .unwrap_or(0);

    let primary_choice = inquire::Select::new(
        "Select primary model:",
        primary_options,
    )
    .with_starting_cursor(default_idx)
    .with_help_message("This is the main reasoning/coding model. Arrow keys to navigate, Enter to select.")
    .with_page_size(10)
    .prompt()?;

    // Parse the selection back to a RecommendedModel
    let primary_name = primary_choice
        .find(" (")
        .map(|i| &primary_choice[..i])
        .unwrap_or(&primary_choice);

    let mut selected_models = Vec::new();

    if let Some(primary) = profile
        .recommended_models
        .iter()
        .find(|m| m.name == primary_name)
    {
        selected_models.push(ModelSelection {
            name: primary.name.clone(),
            quant: Some(primary.best_quant.clone()),
        });

        // ── Step 2: Select the secondary model (dynamic based on remaining VRAM) ──
        let remaining_vram = profile.available_memory_gb - primary.memory_gb;

        // Filter autocomplete models that fit in remaining VRAM
        let secondary_candidates: Vec<&crate::profiling::RecommendedModel> = profile
            .recommended_models
            .iter()
            .filter(|m| {
                m.is_autocomplete
                    && m.memory_gb <= remaining_vram
                    && m.name != primary.name
            })
            .collect();

        if !secondary_candidates.is_empty() {
            let mut secondary_options: Vec<String> = vec!["Skip (single model only)".to_string()];
            secondary_options.extend(secondary_candidates.iter().map(|m| {
                format!(
                    "{} ({}, {:.1} GB, Score: {:.1})",
                    m.name, m.best_quant, m.memory_gb, m.score
                )
            }));

            let secondary_choice = inquire::Select::new(
                &format!(
                    "Select secondary/autocomplete model ({:.1} GB VRAM remaining):",
                    remaining_vram
                ),
                secondary_options,
            )
            .with_starting_cursor(1.min(secondary_candidates.len())) // default to first real model
            .with_help_message("Smaller model for fast sub-agent/autocomplete tasks. 'Skip' for single model.")
            .with_page_size(10)
            .prompt()?;

            if !secondary_choice.starts_with("Skip") {
                let sec_name = secondary_choice
                    .find(" (")
                    .map(|i| &secondary_choice[..i])
                    .unwrap_or(&secondary_choice);

                if let Some(sec) = profile
                    .recommended_models
                    .iter()
                    .find(|m| m.name == sec_name)
                {
                    selected_models.push(ModelSelection {
                        name: sec.name.clone(),
                        quant: Some(sec.best_quant.clone()),
                    });
                }
            }
        } else if remaining_vram > 0.5 {
            println!(
                "  {} No autocomplete models fit in remaining {:.1} GB VRAM — using single model.",
                crate::style("ℹ").blue(),
                remaining_vram
            );
        }
    } else {
        // Fallback for non-dynamic (static model list)
        selected_models.push(ModelSelection {
            name: primary_name.to_string(),
            quant: None,
        });
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
    // OpenCode uses @ai-sdk/openai-compatible which expects the full /v1 base URL.
    let openai_base_url = format!("http://localhost:{}/v1", config.port);
    // Claude Code appends its own /v1/messages path, so we must NOT include /v1 here.
    let anthropic_base_url = format!("http://localhost:{}", config.port);
    let standard_model = config
        .models
        .iter()
        .find(|m| !crate::runner::is_autocomplete_model(&m.name))
        .or_else(|| config.models.first())
        .map(|m| m.name.clone())
        .unwrap_or_else(|| "default".to_string());
    let autocomplete_model = config
        .models
        .iter()
        .find(|m| crate::runner::is_autocomplete_model(&m.name))
        .map(|m| m.name.clone());

    println!(
        "\n{}",
        crate::style("⚙️ Configuration Instructions").bold().cyan()
    );

    println!("\n{}", crate::style("--- OpenCode ---").bold().yellow());
    println!("To use your local server in OpenCode, update your `opencode.json`:");

    println!("{{");
    println!("  \"$schema\": \"https://opencode.ai/config.json\",");
    println!("  \"model\": \"{}\",", standard_model);
    if let Some(ref auto_model) = autocomplete_model {
        println!("  \"small_model\": \"{}\",", auto_model);
    }
    println!("  \"compaction\": {{");
    println!("    \"auto\": true,");
    println!("    \"prune\": true,");
    println!("    \"reserved\": 3000");
    println!("  }},");
    println!("  \"provider\": {{");
    println!("    \"localcode\": {{");
    println!("      \"models\": {{");
    println!("        \"{}\": {{", standard_model);
    println!("          \"name\": \"{}\"", standard_model);
    println!("        }}");
    if let Some(ref auto_model) = autocomplete_model {
        println!("        ,\"{}\": {{", auto_model);
        println!("          \"name\": \"{}\"", auto_model);
        println!("        }}");
    }
    println!("      }},");
    println!("      \"name\": \"LocalCode\",");
    println!("      \"npm\": \"@ai-sdk/openai-compatible\",");
    println!("      \"options\": {{");
    println!("        \"provider\": \"openai\",");
    println!("        \"baseURL\": \"{}\"", openai_base_url);
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
    println!(
        "{} ANTHROPIC_BASE_URL=\"{}\"",
        shell_cmd, anthropic_base_url
    );
    println!("{} ANTHROPIC_API_KEY=\"sk-localcode\"", shell_cmd);

    // Advise on CLAUDE_CODE_MAX_CONTEXT_TOKENS aligned to the model's ctx_size.
    // Claude Code must know the local model's context limit so it can size its
    // system prompt, tool definitions, and conversation history to fit.
    // Formula: max_context_tokens = ctx_size - response_headroom
    let ctx_size = config
        .llama_server_args
        .as_ref()
        .and_then(|a| a.ctx_size)
        .unwrap_or(32768);
    // Reserve ~15% for the model's response, minimum 4096 tokens.
    let response_headroom = std::cmp::max(4096, ctx_size / 7);
    let max_context_tokens = ctx_size.saturating_sub(response_headroom);
    println!(
        "{} CLAUDE_CODE_MAX_CONTEXT_TOKENS={}",
        shell_cmd, max_context_tokens
    );

    println!("claude");

    println!(
        "\n{}",
        crate::style("💡 Context Token Alignment").bold().magenta()
    );
    println!(
        "  Your model's context size is {} tokens.",
        crate::style(ctx_size).yellow()
    );
    println!(
        "  CLAUDE_CODE_MAX_CONTEXT_TOKENS is set to {} (reserves {} tokens for model response).",
        crate::style(max_context_tokens).green(),
        response_headroom
    );
    println!(
        "  If you change ctx_size in localcode.json, re-run `{}` to see updated values.",
        crate::style("localcode info").cyan()
    );
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
            cpu_cores: 16,
            gpu_name: Some("NVIDIA GeForce RTX 5060 Ti".to_string()),
            gpu_backend: llmfit_core::hardware::GpuBackend::Cuda,
            gpu_count: 1,
            unified_memory: false,
            recommended_models: vec![],
            available_memory_gb: 16.0,
        };
        let models = vec![ModelSelection {
            name: "Qwen2.5-Coder-7B-Instruct".to_string(),
            quant: Some("Q8_0".to_string()),
        }];
        let args = LlamaServerArgs::from_hardware(&profile, &models);
        // Qwen 7B: native 32768, capped at native (no YaRN extension).
        // KV q8_0 selected (VRAM headroom > 2 GB). VRAM fits 32768 easily.
        assert_eq!(args.ctx_size, Some(32768));
        assert_eq!(args.n_gpu_layers, Some(999)); // 7B Q8_0 ≈ 7 GB fits in 16 GB
        assert_eq!(args.flash_attn, Some("on".to_string())); // CUDA → flash on
        assert_eq!(args.cache_type_k, Some("q8_0".to_string())); // headroom > 2 GB → q8_0
        assert_eq!(
            args.extra_args.get("slot-save-path").unwrap(),
            &serde_json::json!("/models")
        );
        // Threads: 16 logical cores → 8 physical, GPU mode caps at 8
        assert_eq!(args.extra_args.get("threads").unwrap(), &serde_json::json!(8));
        // Should have multiple parallel slots with the VRAM headroom
        assert!(args.extra_args.get("parallel").is_some());
    }

    #[test]
    fn test_calculate_max_ctx_known_values() {
        // Qwen 7B Q8_0 on 16GB with q8_0 KV cache
        // Capped at native 32768 (no YaRN extension)
        let ctx = LlamaServerArgs::calculate_max_ctx(16.0, 7.0, "Q8_0", "q8_0", "Qwen2.5-Coder-7B-Instruct");
        assert_eq!(ctx, 32768, "7B on 16GB: capped at native 32768");

        // 14B Q4_K_M on 12GB with q4_0 KV cache — capped at native 32768
        let ctx = LlamaServerArgs::calculate_max_ctx(12.0, 14.0, "Q4_K_M", "q4_0", "Qwen2.5-Coder-14B-Instruct");
        assert!(ctx >= 8192, "Expected ≥8192, got {ctx}");

        // 7B Q4_K_M on 8GB with q4_0 KV
        let ctx = LlamaServerArgs::calculate_max_ctx(8.0, 7.0, "Q4_K_M", "q4_0", "Qwen2.5-Coder-7B-Instruct");
        assert!(ctx >= 16384, "Expected ≥16384, got {ctx}");

        // Tiny VRAM — model doesn't fit, returns min(native, 8192)
        let ctx = LlamaServerArgs::calculate_max_ctx(2.0, 7.0, "Q8_0", "q8_0", "Qwen2.5-Coder-7B-Instruct");
        assert!(ctx >= 8192 && ctx <= 32768, "Tiny VRAM ctx should be reasonable, got {ctx}");

        // DeepSeek 7B has 131072 native context — VRAM limits it below that
        let ctx = LlamaServerArgs::calculate_max_ctx(16.0, 7.0, "Q8_0", "q8_0", "DeepSeek-Coder-V2-Lite-7B");
        assert!(ctx >= 49152, "DeepSeek should allow large context, got {ctx}");

        // UD-Q4_K_XL quant — bpp is slightly higher than Q4_K_M so less ctx fits
        let ctx_ud = LlamaServerArgs::calculate_max_ctx(16.0, 14.0, "UD-Q4_K_XL", "q4_0", "Qwen2.5-Coder-14B-Instruct");
        let ctx_std = LlamaServerArgs::calculate_max_ctx(16.0, 14.0, "Q4_K_M", "q4_0", "Qwen2.5-Coder-14B-Instruct");
        // UD-Q4_K_XL uses ~12% more memory for weights, so calculated ctx should be ≤ Q4_K_M's
        assert!(ctx_ud <= ctx_std, "UD ctx {ctx_ud} should be ≤ standard ctx {ctx_std}");
        // Both capped at native 32768 for Qwen 14B
        assert!(ctx_ud >= 8192, "UD ctx {ctx_ud} should be at least 8192");
    }

    #[test]
    fn test_quality_cap_per_model_family() {
        // Qwen 7B: native 32768 (capped here, no YaRN extension)
        assert_eq!(LlamaServerArgs::native_ctx_length("Qwen2.5-Coder-7B-Instruct"), 32768);
        // Qwen 72B: native 131072
        assert_eq!(LlamaServerArgs::native_ctx_length("Qwen2.5-Coder-72B-Instruct"), 131072);
        // Llama 3.1: native 131072
        assert_eq!(LlamaServerArgs::native_ctx_length("Meta-Llama-3.1-8B-Instruct"), 131072);
        // Llama 3 (original): native 8192
        assert_eq!(LlamaServerArgs::native_ctx_length("Meta-Llama-3-8B-Instruct"), 8192);
        // Gemma 2: native 8192
        assert_eq!(LlamaServerArgs::native_ctx_length("gemma-2-9b-it"), 8192);
        // Unknown model → fallback 32768
        assert_eq!(LlamaServerArgs::native_ctx_length("some-random-model"), 32768);
    }

    #[test]
    fn test_parse_params_b() {
        assert_eq!(LlamaServerArgs::parse_params_b("Qwen2.5-Coder-7B-Instruct"), Some(7.0));
        assert_eq!(LlamaServerArgs::parse_params_b("Qwen2.5-Coder-14B-Instruct"), Some(14.0));
        assert_eq!(LlamaServerArgs::parse_params_b("DeepSeek-Coder-V2-Lite-Instruct-1.5B-GGUF"), Some(1.5));
        assert_eq!(LlamaServerArgs::parse_params_b("some-model-no-size"), None);
    }

    #[test]
    fn test_normalize_quant_bpp() {
        // Standard quants pass through unchanged
        let q4_bpp = LlamaServerArgs::normalize_quant_bpp("Q4_K_M");
        assert_eq!(q4_bpp, 0.58);
        assert_eq!(LlamaServerArgs::normalize_quant_bpp("Q8_0"), 1.05);

        // UD- prefix stripped, base quant used
        assert_eq!(LlamaServerArgs::normalize_quant_bpp("UD-Q8_0"), 1.05);

        // _XL suffix → base_M × 1.12
        let ud_q4_xl = LlamaServerArgs::normalize_quant_bpp("UD-Q4_K_XL");
        assert!((ud_q4_xl - 0.58 * 1.12).abs() < 0.001, "UD-Q4_K_XL ≈ 0.6496, got {ud_q4_xl}");

        // _XS suffix → base_M × 0.90
        let ud_q4_xs = LlamaServerArgs::normalize_quant_bpp("UD-Q4_K_XS");
        assert!((ud_q4_xs - 0.58 * 0.90).abs() < 0.001, "UD-Q4_K_XS ≈ 0.522, got {ud_q4_xs}");

        // Non-UD _XL (hypothetical) also works
        let q5_xl = LlamaServerArgs::normalize_quant_bpp("Q5_K_XL");
        assert!((q5_xl - 0.68 * 1.12).abs() < 0.001, "Q5_K_XL ≈ 0.7616, got {q5_xl}");
    }

    #[test]
    fn test_from_hardware_vulkan_no_flash_attn() {
        let profile = HardwareProfile {
            vram_gb: 8.0,
            ram_gb: 16.0,
            cpu_cores: 8,
            gpu_name: Some("AMD Radeon RX 7600".to_string()),
            gpu_backend: llmfit_core::hardware::GpuBackend::Vulkan,
            gpu_count: 1,
            unified_memory: false,
            recommended_models: vec![],
            available_memory_gb: 8.0,
        };
        let models = vec![ModelSelection {
            name: "Qwen2.5-Coder-7B-Instruct".to_string(),
            quant: Some("Q4_K_M".to_string()),
        }];
        let args = LlamaServerArgs::from_hardware(&profile, &models);
        assert_eq!(args.flash_attn, Some("off".to_string())); // Vulkan → no flash
        assert_eq!(args.n_gpu_layers, Some(999)); // 7B Q4_K_M ≈ 4.06 GB fits in 8 GB
    }

    #[test]
    fn test_from_hardware_apple_silicon() {
        let profile = HardwareProfile {
            vram_gb: 16.0, // unified memory reported as GPU
            ram_gb: 16.0,
            cpu_cores: 10,
            gpu_name: Some("Apple M2 Pro".to_string()),
            gpu_backend: llmfit_core::hardware::GpuBackend::Metal,
            gpu_count: 1,
            unified_memory: true,
            recommended_models: vec![],
            available_memory_gb: 16.0,
        };
        let models = vec![ModelSelection {
            name: "Qwen2.5-Coder-7B-Instruct".to_string(),
            quant: Some("Q4_K_M".to_string()),
        }];
        let args = LlamaServerArgs::from_hardware(&profile, &models);
        assert_eq!(args.n_gpu_layers, Some(999)); // Unified memory → always full offload
        assert_eq!(args.flash_attn, Some("on".to_string())); // Metal → flash on
    }

    #[test]
    fn test_from_hardware_partial_offload() {
        // 4GB VRAM with a 7B Q8_0 model (~7.35 GB) → can't fit all layers
        let profile = HardwareProfile {
            vram_gb: 4.0,
            ram_gb: 32.0,
            cpu_cores: 16,
            gpu_name: Some("NVIDIA GeForce GTX 1650".to_string()),
            gpu_backend: llmfit_core::hardware::GpuBackend::Cuda,
            gpu_count: 1,
            unified_memory: false,
            recommended_models: vec![],
            available_memory_gb: 4.0,
        };
        let models = vec![ModelSelection {
            name: "Qwen2.5-Coder-7B-Instruct".to_string(),
            quant: Some("Q8_0".to_string()),
        }];
        let args = LlamaServerArgs::from_hardware(&profile, &models);
        // 4.0 / 7.35 ≈ 54% → 28 layers × 0.54 ≈ 15
        assert!(args.n_gpu_layers.unwrap() > 0);
        assert!(args.n_gpu_layers.unwrap() < 999);
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
    fn test_for_secondary_model() {
        let profile = HardwareProfile {
            vram_gb: 16.0,
            ram_gb: 32.0,
            cpu_cores: 16,
            gpu_name: Some("NVIDIA GeForce RTX 5060 Ti".to_string()),
            gpu_backend: llmfit_core::hardware::GpuBackend::Cuda,
            gpu_count: 1,
            unified_memory: false,
            recommended_models: vec![],
            available_memory_gb: 16.0,
        };
        let primary = LlamaServerArgs {
            ctx_size: Some(65536),
            n_gpu_layers: Some(999),
            flash_attn: Some("on".to_string()),
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("q8_0".to_string()),
            extra_args: {
                let mut e = HashMap::new();
                e.insert("threads".to_string(), serde_json::json!(8));
                e
            },
        };

        // phi-3-mini has native 4096, ctx = max(4096, 8192) = 8192 floor
        let mini = ModelSelection {
            name: "microsoft/phi-3-mini-4k-instruct".to_string(),
            quant: Some("Q8_0".to_string()),
        };
        let secondary = LlamaServerArgs::for_secondary_model(&primary, &mini, &profile);
        assert_eq!(secondary.ctx_size, Some(8192), "phi-3-mini should get 8192 (floor)");
        assert_eq!(secondary.cache_type_k, Some("q4_0".to_string()), "secondary uses q4_0 KV to conserve VRAM");
        assert_eq!(secondary.n_gpu_layers, Some(999), "small model fits in VRAM");
        let cli = secondary.to_cli_args();
        assert!(!cli.contains("--ctx-size 65536"), "Must NOT inherit primary's 65k ctx");
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
