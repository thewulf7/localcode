use crate::ui::ModelSelection;
use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use tokio::process::Command;

/// Qwen-style Jinja chat template embedded at compile time.
/// Kept for backwards compatibility. By default the model's built-in template is used
/// for better tool-call detection across all model families.
#[allow(dead_code)]
const CLAUDE_CODE_JINJA: &str = include_str!("claude_code.jinja");

pub async fn extract_hf_repo_and_file(
    model_name: &str,
    quant: &Option<String>,
) -> (String, Option<String>) {
    if let Some(q) = quant {
        // Dynamic llmfit model — build_gguf_candidates will handle resolution.
        // Return the first candidate here; download_models tries all candidates.
        let parts: Vec<&str> = model_name.split('/').collect();
        let mut base_name = if parts.len() > 1 {
            parts[1]
        } else {
            model_name
        };

        // Strip quantization suffixes that llmfit might include
        if base_name.ends_with("-AWQ") {
            base_name = &base_name[..base_name.len() - 4];
        } else if base_name.ends_with("-GPTQ") || base_name.ends_with("-GGUF") {
            base_name = &base_name[..base_name.len() - 5];
        }

        // UD- (Unsloth Dynamic) quants are published by `unsloth`;
        // standard GGUF quants are published by `bartowski`.
        let repo_org = if q.starts_with("UD-") { "unsloth" } else { "bartowski" };
        let repo = format!("{}/{}-GGUF", repo_org, base_name);
        let file = format!("{}-{}.gguf", base_name, q);

        return (repo, Some(file));
    }

    let default_url = match model_name {
        "llama3-70b-instruct" => "https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf".to_string(),
        "mixtral-8x7b-instruct" => "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
        "llama3-8b-instruct" => "https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf".to_string(),
        "phi3-mini" => "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf".to_string(),
        "gemma-2b-it" => "https://huggingface.co/google/gemma-2b-it-GGUF/resolve/main/2b-it-v1.1-q4_k_m.gguf".to_string(),
        "qwen2-7b-instruct" => "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf".to_string(),
        "mistral-7b-instruct" => "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf".to_string(),
        _ => "https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf".to_string(), // Fallback
    };

    ("".to_string(), Some(default_url))
}

/// Build a list of candidate (repo, file) pairs for a GGUF model download.
/// HuggingFace repo names are case-sensitive — bartowski, the original org, and
/// lmstudio-community may each use slightly different casing or suffixes.
/// The download function tries each candidate in order until one succeeds.
fn build_gguf_candidates(model_name: &str, quant: &str) -> Vec<(String, String)> {
    let parts: Vec<&str> = model_name.split('/').collect();
    let org = if parts.len() > 1 { parts[0] } else { "" };
    let mut base_name = if parts.len() > 1 {
        parts[1]
    } else {
        model_name
    };

    // Strip quantization suffixes
    if base_name.ends_with("-AWQ") {
        base_name = &base_name[..base_name.len() - 4];
    } else if base_name.ends_with("-GPTQ") || base_name.ends_with("-GGUF") {
        base_name = &base_name[..base_name.len() - 5];
    }

    let file = format!("{}-{}.gguf", base_name, quant);
    let is_ud = quant.starts_with("UD-");
    let mut candidates = Vec::new();

    // 1. unsloth — publisher of UD (Unsloth Dynamic) quants; try first for UD-
    if is_ud {
        candidates.push((format!("unsloth/{}-GGUF", base_name), file.clone()));
    }

    // 2. bartowski — the most common GGUF repacker
    candidates.push((format!("bartowski/{}-GGUF", base_name), file.clone()));

    // 3. Original org's own GGUF repo (some publishers have official GGUFs)
    if !org.is_empty() {
        candidates.push((format!("{}/{}-GGUF", org, base_name), file.clone()));
        // Some orgs use lowercase "-gguf" suffix (e.g. microsoft)
        candidates.push((format!("{}/{}-gguf", org, base_name), file.clone()));
    }

    // 4. unsloth fallback — also publishes standard quants for many models
    if !is_ud {
        candidates.push((format!("unsloth/{}-GGUF", base_name), file.clone()));
    }

    // 5. lmstudio-community — another major GGUF publisher
    candidates.push((
        format!("lmstudio-community/{}-GGUF", base_name),
        file,
    ));

    candidates
}

/// Given a `RepoInfo` (from `hf_hub`), find the GGUF file that best matches
/// the requested quantization level.  Official repos often use simplified
/// quant names (e.g. `q4` instead of `Q4_K_M`), so we try several heuristics.
fn find_best_gguf_in_repo(info: &hf_hub::api::RepoInfo, quant: &str) -> Option<String> {
    let gguf_files: Vec<&str> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|f| f.ends_with(".gguf"))
        .collect();

    if gguf_files.is_empty() {
        return None;
    }

    let q_lower = quant.to_lowercase();

    // Build search tokens from the quant string.
    // Q4_K_M    → ["q4_k_m", "q4_k", "q4"]
    // Q8_0      → ["q8_0", "q8"]
    // UD-Q4_K_XL → ["ud-q4_k_xl", "ud-q4_k", "ud-q4", "q4_k_xl", "q4_k", "q4"]
    let mut search_tokens = vec![q_lower.clone()];
    if let Some(pos) = q_lower.rfind('_') {
        search_tokens.push(q_lower[..pos].to_string());
        if let Some(pos2) = q_lower[..pos].rfind('_') {
            search_tokens.push(q_lower[..pos2].to_string());
        }
    }
    // For UD- (Unsloth Dynamic) quants, also add tokens without the UD- prefix
    // so we can match standard quant files as a fallback.
    if q_lower.starts_with("ud-") {
        let stripped = &q_lower[3..];
        if !search_tokens.contains(&stripped.to_string()) {
            search_tokens.push(stripped.to_string());
        }
        if let Some(pos) = stripped.rfind('_') {
            let token = stripped[..pos].to_string();
            if !search_tokens.contains(&token) {
                search_tokens.push(token);
            }
            if let Some(pos2) = stripped[..pos].rfind('_') {
                let token2 = stripped[..pos2].to_string();
                if !search_tokens.contains(&token2) {
                    search_tokens.push(token2);
                }
            }
        }
    }

    // Try each token and return the first file that matches (case-insensitive)
    for token in &search_tokens {
        for f in &gguf_files {
            if f.to_lowercase().contains(token) {
                return Some(f.to_string());
            }
        }
    }

    // No quant match — fall back to the first GGUF file in the repo
    Some(gguf_files[0].to_string())
}

pub async fn download_models(
    models: &[ModelSelection],
    models_dir: &std::path::Path,
) -> Result<std::collections::HashMap<String, std::path::PathBuf>> {
    use console::style;
    use indicatif::{ProgressBar, ProgressStyle};

    // Maps model name → actual local path returned by hf_hub::get().
    // Used by start_llama_swap_docker to generate --model /models/... args
    // without needing to re-discover files via find_local_gguf.
    let mut downloaded_files: std::collections::HashMap<String, std::path::PathBuf> = std::collections::HashMap::new();

    let api = ApiBuilder::new()
        .with_cache_dir(models_dir.to_path_buf())
        .with_token(std::env::var("HF_TOKEN").ok())
        .build()?;

    // Pre-scan: gather all locally available models
    let local_models = crate::models::find_all_local_models(models_dir);

    for m in models {
        let (repo, file) = extract_hf_repo_and_file(&m.name, &m.quant).await;

        if repo.is_empty() || file.is_none() {
            continue;
        }

        let file_name = file.unwrap();

        // Check if this model file already exists locally (by filename match)
        let already_exists = local_models
            .iter()
            .any(|lm| lm.name == file_name || lm.name.to_lowercase() == file_name.to_lowercase());

        if already_exists {
            // Record the path so config generation can use --model directly.
            // Use the actual on-disk name (may differ in case from our constructed name).
            let actual_name = local_models
                .iter()
                .find(|lm| lm.name.to_lowercase() == file_name.to_lowercase())
                .map(|lm| lm.name.clone())
                .unwrap_or_else(|| file_name.clone());
            if let Some(rel) = find_local_gguf(models_dir, &actual_name) {
                downloaded_files.insert(m.name.clone(), models_dir.join(rel.replace('/', std::path::MAIN_SEPARATOR_STR)));
            }
            println!(
                "{} {} {}",
                style("✓").green().bold(),
                style(&m.name).magenta(),
                style("already cached locally, skipping download.").dim()
            );
            continue;
        }

        println!(
            "{} {}",
            style("📥 Checking/Downloading").cyan(),
            style(&m.name).bold().magenta()
        );

        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );

        // For dynamic llmfit models (quant is set), try multiple GGUF sources
        // since HuggingFace repos are case-sensitive and different publishers
        // use different naming conventions and quant labels.
        let mut downloaded = false;
        if m.quant.is_some() {
            let quant_str = m.quant.as_deref().unwrap_or("Q4_K_M");
            let candidates = build_gguf_candidates(&m.name, quant_str);
            for (candidate_repo, candidate_file) in &candidates {
                pb.set_message(format!("Trying {}/{}", candidate_repo, candidate_file));

                // First, try the exact constructed filename
                let repo_c = candidate_repo.clone();
                let file_c = candidate_file.clone();
                let api_c = api.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let repo_api = api_c.model(repo_c);
                    repo_api.get(&file_c)
                })
                .await?;

                if let Ok(path) = result {
                    downloaded_files.insert(m.name.clone(), path);
                    pb.finish_with_message(format!(
                        "✅ {} downloaded from {}",
                        candidate_file, candidate_repo
                    ));
                    downloaded = true;
                    break;
                }

                // Exact filename failed — list repo files and find the best
                // matching GGUF.  Official repos often use different quant
                // naming (e.g. "q4" instead of "Q4_K_M", or capital first letter).
                pb.set_message(format!("Searching {} for GGUF files...", candidate_repo));
                let repo_c2 = candidate_repo.clone();
                let quant_c = quant_str.to_string();
                let api_c2 = api.clone();
                let discovered = tokio::task::spawn_blocking(move || {
                    let repo_api = api_c2.model(repo_c2);
                    match repo_api.info() {
                        Ok(info) => find_best_gguf_in_repo(&info, &quant_c),
                        Err(_) => None, // repo doesn't exist, skip
                    }
                })
                .await?;

                if let Some(real_file) = discovered {
                    pb.set_message(format!("Downloading {}", real_file));
                    let repo_c3 = candidate_repo.clone();
                    let file_c3 = real_file.clone();
                    let api_c3 = api.clone();
                    let result2 = tokio::task::spawn_blocking(move || {
                        let repo_api = api_c3.model(repo_c3);
                        repo_api.get(&file_c3)
                    })
                    .await?;

                    if let Ok(path) = result2 {
                        downloaded_files.insert(m.name.clone(), path);
                        pb.finish_with_message(format!(
                            "✅ {} downloaded from {}",
                            real_file, candidate_repo
                        ));
                        downloaded = true;
                        break;
                    }
                }
            }
        } else {
            // Legacy static model — single repo from extract_hf_repo_and_file
            pb.set_message(format!("Downloading {}", file_name));
            let repo_clone = repo.clone();
            let file_name_clone = file_name.clone();
            let api_clone = api.clone();
            let result = tokio::task::spawn_blocking(move || {
                let repo_api = api_clone.model(repo_clone);
                repo_api.get(&file_name_clone)
            })
            .await?;

            if let Ok(path) = result {
                downloaded_files.insert(m.name.clone(), path);
                pb.finish_with_message(format!("✅ {} downloaded.", file_name));
                downloaded = true;
            }
        }

        if !downloaded {
            pb.finish_with_message(format!(
                "⚠️  Could not find GGUF for {} in any known repository",
                m.name
            ));
        }
    }

    Ok(downloaded_files)
}

/// Walk `models_dir` looking for a `.gguf` file whose name matches `filename`.
/// Returns the path relative to `models_dir` using forward-slashes (for Docker).
/// Comparison is case-insensitive — HF repos may use different casing than the
/// filename we construct (e.g. "Phi-3-..." vs "phi-3-...").
fn find_local_gguf(models_dir: &std::path::Path, filename: &str) -> Option<String> {
    use walkdir::WalkDir;
    let needle = filename.to_lowercase();
    for entry in WalkDir::new(models_dir).into_iter().filter_map(|e| e.ok()) {
        if entry.path().is_file() {
            let is_match = entry
                .path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.to_lowercase() == needle)
                .unwrap_or(false);
            if is_match && let Ok(rel) = entry.path().strip_prefix(models_dir) {
                // Use forward slashes for the Linux container path
                return Some(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    None
}

pub async fn start_llama_swap_docker(
    models: &[ModelSelection],
    models_dir: &std::path::Path,
    port: u16,
    llama_server_args: Option<&crate::ui::LlamaServerArgs>,
    downloaded_files: &std::collections::HashMap<String, std::path::PathBuf>,
    profile: Option<&crate::profiling::HardwareProfile>,
) -> Result<()> {
    println!("📦 Launching localcode container...");

    // First, verify docker is installed
    let docker_check = Command::new("docker")
        .arg("--version")
        .output()
        .await
        .context("Failed to execute docker command. Is docker installed?")?;

    if !docker_check.status.success() {
        return Err(anyhow::anyhow!(
            "Docker is not running or not installed correctly: {}",
            String::from_utf8_lossy(&docker_check.stderr)
        ));
    }

    // Attempt to forcefully remove any existing container with the same name to avoid conflicts
    let _ = Command::new("docker")
        .args(["rm", "-f", "localcode-llm"])
        .output()
        .await;

    // Generate config.yaml for llama-swap
    let mut yaml_content =
        String::from("includeAliasesInList: true\nsendLoadingState: true\n\nmodels:\n");
    let mut autocomplete_models = Vec::new();
    let mut autocomplete_model_name: Option<String> = None;

    // Pre-scan: detect if any non-primary model qualifies as autocomplete/small.
    // We need this BEFORE generating YAML so the primary model knows whether to
    // claim haiku aliases or leave them for the small model.
    let has_small_model = models.iter().skip(1).any(|m| is_autocomplete_model(&m.name));

    let mut assigned_aliases = false;
    for m in models {
        let (repo, file) = extract_hf_repo_and_file(&m.name, &m.quant).await;

        // Quote the model name key to handle slashes or special chars safely in YAML
        yaml_content.push_str(&format!("  \"{}\":\n", m.name));

        let is_autocomplete = is_autocomplete_model(&m.name);

        // Only add to autocomplete group if it's NOT the primary model (assigned_aliases is true after the first)
        if assigned_aliases && is_autocomplete {
            autocomplete_models.push(m.name.clone());
            if autocomplete_model_name.is_none() {
                autocomplete_model_name = Some(m.name.clone());
            }
        }

        // Prefer the actual path returned by download_models (hf_hub::get()).
        // This avoids filename mismatch issues from fuzzy matching (e.g.
        // "Phi-3-mini-4k-instruct-q4.gguf" vs "phi-3-...-Q4_K_M.gguf").
        let source_args = if let Some(local_path) = downloaded_files.get(&m.name) {
            // Convert host path to Docker-relative /models/... path
            let rel = local_path
                .strip_prefix(models_dir)
                .unwrap_or(local_path.as_path());
            let docker_path = rel.to_string_lossy().replace('\\', "/");
            format!("--model /models/{}", docker_path)
        } else if let Some(ref f) = file {
            // Fallback: try to find the canonical filename locally
            if let Some(rel_path) = find_local_gguf(models_dir, f) {
                format!("--model /models/{}", rel_path)
            } else {
                // Last resort: HF download at runtime (requires SSL in llama-server)
                let repo_part = if !repo.is_empty() {
                    format!("--hf-repo {}", repo)
                } else {
                    String::new()
                };
                format!("{} --hf-file {}", repo_part, f)
            }
        } else if !repo.is_empty() {
            format!("--hf-repo {}", repo)
        } else {
            String::new()
        };

        let mut custom_args = if assigned_aliases {
            // Secondary/autocomplete model: compute lighter per-model args with
            // ctx_size capped to the model's native training context.
            if let (Some(primary), Some(prof)) = (llama_server_args, profile) {
                let secondary_args = crate::ui::LlamaServerArgs::for_secondary_model(primary, m, prof);
                secondary_args.to_cli_args()
            } else {
                llama_server_args
                    .map(|a| a.to_cli_args())
                    .unwrap_or_else(|| "--ctx-size 32768".to_string())
            }
        } else {
            // Primary model: use the pre-computed args from from_hardware()
            llama_server_args
                .map(|a| a.to_cli_args())
                .unwrap_or_else(|| "--ctx-size 32768".to_string())
        };

        // Sanitize Windows absolute paths for Docker.
        // If an arg contains a path like C:\Users\..., rewrite it to use /models/ relative to the container.
        if custom_args.contains(':') && custom_args.contains('\\') {
            let expanded_models_dir =
                shellexpand::tilde(models_dir.to_str().unwrap_or("")).to_string();
            // This is a naive but effective replacement for common local paths mapped to /models
            custom_args = custom_args
                .replace(&expanded_models_dir, "/models")
                .replace('\\', "/");
        }

        if !assigned_aliases {
            assigned_aliases = true;
            yaml_content.push_str(&format!(
                // Let each model use its OWN built-in chat template for tool calling.
                // The native template is stored inside the GGUF and llama.cpp knows how
                // to parse its tool-call format (Qwen <tool_call>, Llama <|python_tag|>, etc.).
                // Overriding with a custom template breaks non-Qwen models because they
                // don't generate the Qwen-specific <tool_call> XML tags.
                //
                // --reasoning-format none: prevents llama-server from incorrectly
                // auto-detecting a reasoning format (e.g. "deepseek" for Qwen) which
                // disrupts grammar-constrained tool-call generation.
                //
                // Context is capped at the model's native training length
                // (e.g. 32768 for Qwen2.5-7B).  YaRN rope scaling was removed
                // because extending context beyond the training window causes
                // attention degradation that produces gibberish — especially on
                // ≤14B models doing structured tool-call generation.
                "    cmd: llama-server --port ${{PORT}} {} --host 0.0.0.0 --jinja --reasoning-format none {}\n",
                source_args,
                custom_args
            ));
            // strip_params: prevent Claude Code from overriding local model's
            // sampling settings (temperature, top_k, etc.) which degrades quality.
            //
            // setParams tool_choice: force llama-server's grammar-constrained
            // tool-call generation to be active from the first token. Without this,
            // the grammar is "lazy" — it only triggers when the model starts with
            // `<tool_call>\n`. Small models often start with markdown/XML instead,
            // bypassing the grammar entirely and producing raw text instead of
            // structured tool_use blocks. The Anthropic-format object
            // `{"type":"any"}` is converted to OpenAI `"required"` internally.
            yaml_content.push_str("    filters:\n");
            yaml_content
                .push_str("      strip_params: \"temperature, top_k, top_p, repeat_penalty\"\n");
            yaml_content.push_str("      setParams:\n");
            yaml_content.push_str("        tool_choice: \"auto\"\n");
            yaml_content.push_str("    aliases:\n");
            // Claude 3.5 series — sonnet/opus → primary model
            yaml_content.push_str("      - \"claude-3-5-sonnet-20241022\"\n");
            yaml_content.push_str("      - \"claude-3-5-sonnet-latest\"\n");
            yaml_content.push_str("      - \"claude-3-opus-20240229\"\n");
            yaml_content.push_str("      - \"claude-3-sonnet-20240229\"\n");
            // Claude 4 series — sonnet/opus → primary model
            yaml_content.push_str("      - \"claude-sonnet-4-5\"\n");
            yaml_content.push_str("      - \"claude-sonnet-4-5-20250929\"\n");
            yaml_content.push_str("      - \"claude-sonnet-4-6\"\n");
            yaml_content.push_str("      - \"claude-sonnet-4-latest\"\n");
            yaml_content.push_str("      - \"claude-opus-4-5\"\n");
            yaml_content.push_str("      - \"claude-opus-4-5-20251101\"\n");
            // If there's NO small model, haiku aliases also go to primary
            if !has_small_model {
                yaml_content.push_str("      - \"claude-3-5-haiku-20241022\"\n");
                yaml_content.push_str("      - \"claude-3-5-haiku-latest\"\n");
                yaml_content.push_str("      - \"claude-3-haiku-20240307\"\n");
                yaml_content.push_str("      - \"claude-haiku-4-5\"\n");
                yaml_content.push_str("      - \"claude-haiku-4-5-20251001\"\n");
            }
        } else {
            yaml_content.push_str(&format!(
                // Secondary/autocomplete models.
                "    cmd: llama-server --port ${{PORT}} {} --host 0.0.0.0 --jinja --reasoning-format none {}\n",
                source_args,
                custom_args
            ));
            // Same filters for the small model.
            yaml_content.push_str("    filters:\n");
            yaml_content
                .push_str("      strip_params: \"temperature, top_k, top_p, repeat_penalty, frequency_penalty, presence_penalty\"\n");
            yaml_content.push_str("      setParams:\n");
            yaml_content.push_str("        max_tokens: 2048\n");
            yaml_content.push_str("        temperature: 0\n");
            yaml_content.push_str("        top_p: 1.0\n");
            yaml_content.push_str("        repeat_penalty: 1.3\n");
            yaml_content.push_str("        frequency_penalty: 0.5\n");
            yaml_content.push_str("        presence_penalty: 0.3\n");
            yaml_content.push_str("        tool_choice: \"auto\"\n");
            // Haiku aliases → small model for subagent routing
            if is_autocomplete {
                yaml_content.push_str("    aliases:\n");
                yaml_content.push_str("      - \"claude-3-5-haiku-20241022\"\n");
                yaml_content.push_str("      - \"claude-3-5-haiku-latest\"\n");
                yaml_content.push_str("      - \"claude-3-haiku-20240307\"\n");
                yaml_content.push_str("      - \"claude-haiku-4-5\"\n");
                yaml_content.push_str("      - \"claude-haiku-4-5-20251001\"\n");
            }
        }
    }

    if !autocomplete_models.is_empty() {
        yaml_content.push_str("\ngroups:\n  autocomplete:\n    persistent: true\n    swap: false\n    exclusive: false\n    members:\n");
        for model_name in autocomplete_models {
            yaml_content.push_str(&format!("      - {}\n", model_name));
        }
    }

    yaml_content.push_str("\nhooks:\n  on_startup:\n    preload:\n");
    for m in models {
        yaml_content.push_str(&format!("      - {}\n", m.name));
    }

    let config_path = models_dir.join("llama-swap.yaml");
    tokio::fs::write(&config_path, yaml_content).await?;

    let port_mapping = format!("{}:8080", port);
    let volume_mapping = format!("{}:/models", models_dir.to_string_lossy());
    let config_mount = format!("{}:/app/config.yaml", config_path.to_string_lossy());

    let mut args = vec![
        "run".to_string(),
        "-d".to_string(), // run completely detached in the background
        "--name".to_string(),
        "localcode-llm".to_string(),
        "--gpus".to_string(),
        "all".to_string(),
        "-e".to_string(),
        "HF_HOME=/models".to_string(),
        "-p".to_string(),
        port_mapping,
        "-v".to_string(),
        volume_mapping,
        "-v".to_string(),
        config_mount,
        "ghcr.io/thewulf7/localcode:cuda-latest".to_string(),
    ];

    let mut output = Command::new("docker").args(&args).output().await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Auto-Detect if the failure is just because they don't have Nvidia Container Toolkit or WSL GPU passthrough set up
        if stderr.contains("could not select device driver") || stderr.contains("nvidia") {
            use console::style;
            println!(
                "{} {}",
                style("⚠️").yellow(),
                style("NVIDIA Container Toolkit not detected or GPU not available.").yellow()
            );
            println!(
                "{} {}",
                style("ℹ").cyan(),
                style("Falling back to CPU mode (this will be slower).").dim()
            );
            println!(
                "  {}",
                style("To enable GPU acceleration, install the NVIDIA Container Toolkit:").dim()
            );
            println!("  {}", style("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html").dim().underlined());
            println!();

            // Re-run without --gpus all and use cpu image
            if let Some(pos) = args.iter().position(|x| x == "--gpus") {
                args.remove(pos); // remove "--gpus"
                args.remove(pos); // remove "all"
            }
            if let Some(pos) = args
                .iter()
                .position(|x| x == "ghcr.io/thewulf7/localcode:cuda-latest")
            {
                args[pos] = "ghcr.io/mostlygeek/llama-swap:cpu".to_string();
            }

            output = Command::new("docker").args(&args).output().await?;

            if !output.status.success() {
                return Err(anyhow::anyhow!(
                    "Docker failed to start container on CPU fallback. Ensure ports are not in use.\nError: {}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        } else {
            return Err(anyhow::anyhow!(
                "Docker failed to start container. Ensure ports are not in use.\nError: {}",
                stderr
            ));
        }
    }

    Ok(())
}

pub async fn show_status() -> Result<()> {
    use console::style;

    println!(
        "{}",
        style("Streaming live logs from localcode-llm container... (Press Ctrl+C to stop)").cyan()
    );

    // We use `--tail 50` so we don't stream gigantic past histories immediately
    let mut child = Command::new("docker")
        .args(["logs", "-f", "--tail", "50", "localcode-llm"])
        .spawn()?;

    tokio::select! {
        _ = child.wait() => {}
        _ = tokio::signal::ctrl_c() => {
            let _ = child.kill().await;
        }
    }

    Ok(())
}

pub async fn stop_server() -> Result<()> {
    use console::style;

    println!(
        "{}",
        style("🛑 Stopping and removing local LLM container (models are preserved on disk)...")
            .yellow()
    );

    let status = Command::new("docker")
        .args(["rm", "-f", "localcode-llm"])
        .output()
        .await?;

    if status.status.success() {
        println!(
            "{} {}",
            style("✓").green().bold(),
            style("Server stopped and container removed.").green()
        );
    }

    Ok(())
}

// Ensure the helper grouping heuristic is standalone so we can cleanly test it
pub fn is_autocomplete_model(model_name: &str) -> bool {
    let lower = model_name.to_lowercase();
    lower.contains("mini")
        || lower.contains("1.5b")
        || lower.contains("2b")
        || lower.contains("0.5b")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extract_hf_repo_and_file_static() {
        let (repo, file) = extract_hf_repo_and_file("phi3-mini", &None).await;
        assert_eq!(repo, "");
        assert_eq!(file, Some("https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf".to_string()));
    }

    #[tokio::test]
    async fn test_extract_hf_repo_and_file_dynamic() {
        let quant = Some("Q4_K_M".to_string());
        // Dynamic llmfit model case
        let (repo, file) = extract_hf_repo_and_file("author/llama3-8b-instruct", &quant).await;
        assert_eq!(repo, "bartowski/llama3-8b-instruct-GGUF");
        assert_eq!(file, Some("llama3-8b-instruct-Q4_K_M.gguf".to_string()));

        // Edge case: single name passed incorrectly
        let (repo2, file2) = extract_hf_repo_and_file("some-custom-model", &quant).await;
        assert_eq!(repo2, "bartowski/some-custom-model-GGUF");
        assert_eq!(file2, Some("some-custom-model-Q4_K_M.gguf".to_string()));
    }

    #[test]
    fn test_build_gguf_candidates() {
        // Standard quant: bartowski first, then original org, unsloth, lmstudio-community
        let candidates = build_gguf_candidates("microsoft/phi-3-mini-4k-instruct", "Q4_K_M");
        assert_eq!(candidates.len(), 5);
        assert_eq!(candidates[0].0, "bartowski/phi-3-mini-4k-instruct-GGUF");
        assert_eq!(candidates[1].0, "microsoft/phi-3-mini-4k-instruct-GGUF");
        assert_eq!(candidates[2].0, "microsoft/phi-3-mini-4k-instruct-gguf");
        assert_eq!(candidates[3].0, "unsloth/phi-3-mini-4k-instruct-GGUF");
        assert_eq!(candidates[4].0, "lmstudio-community/phi-3-mini-4k-instruct-GGUF");
        for (_, f) in &candidates {
            assert_eq!(f, "phi-3-mini-4k-instruct-Q4_K_M.gguf");
        }

        // UD quant: unsloth first, then bartowski, org, lmstudio-community
        let ud_candidates = build_gguf_candidates("microsoft/phi-3-mini-4k-instruct", "UD-Q4_K_XL");
        assert_eq!(ud_candidates[0].0, "unsloth/phi-3-mini-4k-instruct-GGUF");
        assert_eq!(ud_candidates[1].0, "bartowski/phi-3-mini-4k-instruct-GGUF");
        for (_, f) in &ud_candidates {
            assert_eq!(f, "phi-3-mini-4k-instruct-UD-Q4_K_XL.gguf");
        }

        // Without org prefix — bartowski, unsloth, lmstudio-community
        let candidates2 = build_gguf_candidates("some-model", "Q8_0");
        assert_eq!(candidates2.len(), 3);
        assert_eq!(candidates2[0].0, "bartowski/some-model-GGUF");
        assert_eq!(candidates2[1].0, "unsloth/some-model-GGUF");
        assert_eq!(candidates2[2].0, "lmstudio-community/some-model-GGUF");
    }

    #[tokio::test]
    async fn test_extract_hf_repo_and_file_ud_quant() {
        // UD quants should route to unsloth repo
        let quant = Some("UD-Q4_K_XL".to_string());
        let (repo, file) = extract_hf_repo_and_file("Qwen/Qwen2.5-Coder-14B-Instruct", &quant).await;
        assert_eq!(repo, "unsloth/Qwen2.5-Coder-14B-Instruct-GGUF");
        assert_eq!(file, Some("Qwen2.5-Coder-14B-Instruct-UD-Q4_K_XL.gguf".to_string()));

        // Standard quant should still use bartowski
        let quant_std = Some("Q4_K_M".to_string());
        let (repo2, _) = extract_hf_repo_and_file("Qwen/Qwen2.5-Coder-14B-Instruct", &quant_std).await;
        assert_eq!(repo2, "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF");
    }

    #[test]
    fn test_is_autocomplete_model() {
        assert!(is_autocomplete_model("phi3-mini"));
        assert!(is_autocomplete_model("qwen2.5-coder-1.5b-instruct"));
        assert!(is_autocomplete_model("gemma-2b-it"));
        assert!(is_autocomplete_model("some-0.5b-model"));
        assert!(!is_autocomplete_model("llama3-8b-instruct"));
        assert!(!is_autocomplete_model("mixtral-8x7b-instruct"));
        assert!(!is_autocomplete_model("llama3-70b-instruct"));
    }
}
