use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::process::Command;

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

pub async fn start_llama_cpp_docker(repo: &str, models_dir: &std::path::Path, file: Option<&str>, port: u16) -> Result<()> {
    println!("📦 Pulling ghcr.io/ggml-org/llama.cpp:server... (This may take a moment)");
    
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
    
    let port_mapping = format!("{}:8080", port);
    let volume_mapping = format!("{}:/models", models_dir.to_string_lossy());
    let mut args = vec![
        "run".to_string(), 
        "-d".to_string(), // run completely detached in the background
        "--name".to_string(), "opencode-llm".to_string(),
        "--gpus".to_string(), "all".to_string(),
        "-p".to_string(), port_mapping, 
        "-v".to_string(), volume_mapping,
        "ghcr.io/ggml-org/llama.cpp:server".to_string(),
    ];
    
    if !repo.is_empty() {
        args.push("--hf-repo".to_string());
        args.push(repo.to_string());
    }
    
    if let Some(f) = file {
        args.push("--hf-file".to_string());
        args.push(f.to_string());
    }
    
    args.extend(vec![
        "-m".to_string(), "/models/model.gguf".to_string(),
        "--ctx-size".to_string(), "8192".to_string(),
        "--host".to_string(), "0.0.0.0".to_string(),
        "--port".to_string(), "8080".to_string()
    ]);
    
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

            // Re-run without --gpus all
            if let Some(pos) = args.iter().position(|x| x == "--gpus") {
                args.remove(pos); // remove "--gpus"
                args.remove(pos); // remove "all"
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
    use indicatif::{ProgressBar, ProgressStyle};
    use console::style;

    // We use `--tail 50` so we don't stream gigantic past histories immediately
    let mut child = Command::new("docker")
        .args(&["logs", "-f", "--tail", "50", "opencode-llm"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Setup beautiful progress spinner
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
        .template("{spinner:.green} {msg}")
        .unwrap());
    
    let stderr = child.stderr.take().expect("Failed to grab stderr");
    let mut reader = tokio::io::BufReader::new(stderr);
    let mut line = String::new();
    
    pb.set_message("Waiting for container startup logs...");
    
    use tokio::io::AsyncBufReadExt;
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                let trimmed = line.trim();
                // Check if the server is ready
                if trimmed.contains("HTTP server listening") {
                    pb.finish_and_clear();
                    println!("{} {}", style("✅ llama.cpp server is actively running on:").green().bold(), style("http://localhost:8080").cyan());
                    break;
                }
                
                if trimmed.contains("llm_load_print_meta:") {
                    if trimmed.contains("model type =") || trimmed.contains("n_ctx_train =") {
                        let parts: Vec<&str> = trimmed.split("=").collect();
                        if parts.len() == 2 {
                            let stat = parts[0].replace("llm_load_print_meta:", "").trim().to_string();
                            let val = parts[1].trim();
                            pb.println(format!("📊 {}: {}", style(stat).dim(), style(val).yellow()));
                        }
                    }
                }
                
                // Show download progress or loading 
                if trimmed.contains("downloading") {
                    pb.set_message(format!("Downloading model partial over network..."));
                    pb.tick();
                } else if trimmed.contains("llama_model_load") {
                    pb.set_message("Loading buffers into memory...");
                    pb.tick();
                } else if trimmed.contains("ggml_") {
                    pb.set_message("Processing architecture layers...");
                    pb.tick();
                } else if trimmed.contains("llama_kv_cache_init:") {
                    pb.set_message("Calculating KV cache memory blocks...");
                } else if !trimmed.is_empty() {
                    pb.set_message(format!("Status: {:?}", trimmed.chars().take(40).collect::<String>()));
                    pb.tick();
                }
            }
            Err(_) => break,
        }
    }

    tokio::spawn(async move {
        loop {
            line.clear();
            if let Ok(bytes) = reader.read_line(&mut line).await {
                if bytes == 0 { break; }
                let trimmed = line.trim();
                if !trimmed.is_empty() && trimmed.contains("llama_print_timings") {
                    println!("ℹ️ {}", style(trimmed).dim());
                }
            } else {
                break;
            }
        }
    });

    // Wait until user stops watching the logs implicitly by killing status command
    tokio::signal::ctrl_c().await?;

    // the status command drops, child is implicitly aborted locally, but Docker process itself remains attached independently inside the host daemon
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
    } else {
        println!("{} {}", style("ℹ").cyan().bold(), style("No running server found.").dim());
    }
    
    Ok(())
}
