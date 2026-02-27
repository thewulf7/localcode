use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::process::Command;
use crate::ui::ModelSelection;
use hf_hub::api::sync::ApiBuilder;

pub async fn extract_hf_repo_and_file(model_name: &str, quant: &Option<String>) -> (String, Option<String>) {
    if let Some(q) = quant {
        // It's a dynamic llmfit model, format as `user/model` and `*quant.gguf`
        let parts: Vec<&str> = model_name.split('/').collect();
        let base_name = if parts.len() > 1 { parts[1] } else { model_name };
        
        let repo = format!("bartowski/{}-GGUF", base_name);
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

pub async fn download_models(models: &[ModelSelection], models_dir: &std::path::Path) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use console::style;

    let api = ApiBuilder::new()
        .with_cache_dir(models_dir.to_path_buf())
        .build()?;

    for m in models {
        let (repo, file) = extract_hf_repo_and_file(&m.name, &m.quant).await;
        
        if repo.is_empty() || file.is_none() {
            continue;
        }

        let file_name = file.unwrap();

        println!("{} {}", style("📥 Checking/Downloading").cyan(), style(&m.name).bold().magenta());

        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
            .template("{spinner:.green} {msg}")
            .unwrap());
        pb.set_message(format!("Downloading {}", file_name));

        // Use spawn_blocking since hf_hub is sync
        let repo_clone = repo.clone();
        let file_name_clone = file_name.clone();
        let api_clone = api.clone();
        tokio::task::spawn_blocking(move || {
            let repo_api = api_clone.model(repo_clone);
            // This will block until downloaded or verify it exists
            repo_api.get(&file_name_clone)
        }).await??;

        pb.finish_with_message(format!("✅ {} downloaded.", file_name));
    }
    
    Ok(())
}

pub async fn start_llama_swap_docker(models: &[ModelSelection], models_dir: &std::path::Path, port: u16) -> Result<()> {
    println!("📦 Pulling ghcr.io/mostlygeek/llama-swap:cuda... (This may take a moment)");
    
    // First, verify docker is installed
    let docker_check = Command::new("docker")
        .arg("--version")
        .output()
        .await
        .context("Failed to execute docker command. Is docker installed?")?;
        
    if !docker_check.status.success() {
        return Err(anyhow::anyhow!("Docker is not running or not installed correctly: {}", String::from_utf8_lossy(&docker_check.stderr)));
    }

    // Attempt to forcefully remove any existing container with the same name to avoid conflicts
    let _ = Command::new("docker")
        .args(&["rm", "-f", "opencode-llm"])
        .output()
        .await;

    // Generate config.yaml for llama-swap
    let mut yaml_content = String::from("models:\n");
    let mut autocomplete_models = Vec::new();

    for m in models {
        let (repo, file) = extract_hf_repo_and_file(&m.name, &m.quant).await;
        
        yaml_content.push_str(&format!("  {}:\n", m.name));
        
        let is_autocomplete = is_autocomplete_model(&m.name);
        
        if is_autocomplete {
            autocomplete_models.push(m.name.clone());
        }

        let file_arg = if let Some(f) = file {
            format!("--hf-file {}", f)
        } else {
            String::new()
        };
        
        let repo_arg = if !repo.is_empty() {
            format!("--hf-repo {}", repo)
        } else {
            String::new()
        };

        yaml_content.push_str(&format!("    cmd: llama-server --port ${{PORT}} {} {} --host 0.0.0.0 --ctx-size 8192\n", repo_arg, file_arg));
    }

    if !autocomplete_models.is_empty() {
        yaml_content.push_str("\ngroups:\n  autocomplete:\n    persistent: true\n    swap: false\n    exclusive: false\n    members:\n");
        for model_name in autocomplete_models {
            yaml_content.push_str(&format!("      - {}\n", model_name));
        }
    }

    let config_path = models_dir.join("llama-swap.yaml");
    tokio::fs::write(&config_path, yaml_content).await?;
    
    let port_mapping = format!("{}:8080", port);
    let volume_mapping = format!("{}:/models", models_dir.to_string_lossy());
    let config_mount = format!("{}:/app/config.yaml", config_path.to_string_lossy());

    let mut args = vec![
        "run".to_string(), 
        "-d".to_string(), // run completely detached in the background
        "--name".to_string(), "opencode-llm".to_string(),
        "--gpus".to_string(), "all".to_string(),
        "-e".to_string(), "HF_HOME=/models".to_string(),
        "-p".to_string(), port_mapping, 
        "-v".to_string(), volume_mapping,
        "-v".to_string(), config_mount,
        "ghcr.io/mostlygeek/llama-swap:cuda".to_string(),
    ];
    
    let mut output = Command::new("docker")
        .args(&args)
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // Auto-Detect if the failure is just because they don't have Nvidia Container Toolkit or WSL GPU passthrough set up
        if stderr.contains("could not select device driver") || stderr.contains("nvidia") {
            use console::style;
            println!("{} {}", style("⚠️").yellow(), style("NVIDIA Container Toolkit not detected or GPU not available.").yellow());
            println!("{} {}", style("ℹ").cyan(), style("Falling back to CPU mode (this will be slower).").dim());
            println!("  {}", style("To enable GPU acceleration, install the NVIDIA Container Toolkit:").dim());
            println!("  {}", style("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html").dim().underlined());
            println!();

            // Re-run without --gpus all and use cpu image
            if let Some(pos) = args.iter().position(|x| x == "--gpus") {
                args.remove(pos); // remove "--gpus"
                args.remove(pos); // remove "all"
            }
            if let Some(pos) = args.iter().position(|x| x == "ghcr.io/mostlygeek/llama-swap:cuda") {
                args[pos] = "ghcr.io/mostlygeek/llama-swap:cpu".to_string();
            }
            
            output = Command::new("docker")
                .args(&args)
                .output()
                .await?;

            if !output.status.success() {
                return Err(anyhow::anyhow!("Docker failed to start container on CPU fallback. Ensure ports are not in use.\nError: {}", String::from_utf8_lossy(&output.stderr)));
            }
        } else {
            return Err(anyhow::anyhow!("Docker failed to start container. Ensure ports are not in use.\nError: {}", stderr));
        }
    }

    Ok(())
}

pub async fn show_status() -> Result<()> {
    use console::style;

    println!("{}", style("Streaming live logs from opencode-llm container... (Press Ctrl+C to stop)").cyan());

    // We use `--tail 50` so we don't stream gigantic past histories immediately
    let mut child = Command::new("docker")
        .args(&["logs", "-f", "--tail", "50", "opencode-llm"])
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
    
    println!("{}", style("🛑 Stopping and removing local LLM Docker container...").yellow());
    
    let status = Command::new("docker")
        .args(&["rm", "-f", "opencode-llm"])
        .output()
        .await?;

    if status.status.success() {
        println!("{} {}", style("✓").green().bold(), style("Server stopped successfully.").green());
    }
    
    Ok(())
}

// Ensure the helper grouping heuristic is standalone so we can cleanly test it
pub fn is_autocomplete_model(model_name: &str) -> bool {
    let lower = model_name.to_lowercase();
    lower.contains("mini") || 
    lower.contains("coder") || 
    lower.contains("1.5b") || 
    lower.contains("2b") ||
    lower.contains("0.5b")
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
