use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    pub name: String,
    #[allow(dead_code)]
    pub path: PathBuf,
    pub size_bytes: u64,
    pub source: String, // e.g. "LM Studio", "Ollama", "LocalCode"
}

/// Recursively scans a given directory for `.gguf` files.
pub fn scan_directory_for_gguf(dir: &Path, source_name: &str) -> Vec<DiscoveredModel> {
    let mut found = Vec::new();
    if !dir.exists() {
        return found;
    }

    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file()
            && let Some(ext) = path.extension()
            && ext == "gguf"
            && let Ok(metadata) = fs::metadata(path)
            && let Some(file_name) = path.file_name().and_then(|n| n.to_str())
        {
            found.push(DiscoveredModel {
                name: file_name.to_string(),
                path: path.to_path_buf(),
                size_bytes: metadata.len(),
                source: source_name.to_string(),
            });
        }
    }
    found
}

/// Scans standard Ollama manifest structures to locate valid GGUF blobs.
pub fn scan_ollama_cache() -> Vec<DiscoveredModel> {
    let mut found = Vec::new();

    // Resolve ~/.ollama
    let ollama_dir = match shellexpand::tilde("~/.ollama") {
        std::borrow::Cow::Borrowed(s) => PathBuf::from(s),
        std::borrow::Cow::Owned(s) => PathBuf::from(s),
    };

    if !ollama_dir.exists() {
        return found;
    }

    let manifests_dir = ollama_dir
        .join("models")
        .join("manifests")
        .join("registry.ollama.ai");
    let blobs_dir = ollama_dir.join("models").join("blobs");

    if !manifests_dir.exists() || !blobs_dir.exists() {
        return found;
    }

    // Traverse all JSON manifests recursively
    for entry in WalkDir::new(manifests_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file()
            && let Ok(content) = fs::read_to_string(path)
            && let Ok(json) = serde_json::from_str::<Value>(&content)
        {
            // Look for layers
            if let Some(layers) = json.get("layers").and_then(|l| l.as_array()) {
                for layer in layers {
                    if let Some(media_type) = layer.get("mediaType").and_then(|m| m.as_str())
                        && media_type == "application/vnd.ollama.image.model"
                        && let Some(digest) = layer.get("digest").and_then(|d| d.as_str())
                    {
                        // The blob is named exactly after the digest but with sha256- prefix
                        // Some systems use `:` separator, Ollama locally uses `-`
                        let blob_name = digest.replace(":", "-");
                        let blob_path = blobs_dir.join(&blob_name);

                        if blob_path.exists()
                            && let Ok(metadata) = fs::metadata(&blob_path)
                        {
                            // Extract model name from the directory structure if possible
                            // e.g. ~/.ollama/models/manifests/registry.ollama.ai/library/llama3/8b/latest
                            let mut model_name = "Ollama-Model".to_string();
                            if let Some(parent) = path.parent()
                                && let Some(repo_name) = parent
                                    .parent()
                                    .and_then(|p| p.file_name())
                                    .and_then(|f| f.to_str())
                                && let Some(tag_name) = parent.file_name().and_then(|f| f.to_str())
                            {
                                let final_tag = path
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_str()
                                    .unwrap_or("latest");
                                model_name = format!("{}:{}-{}", repo_name, tag_name, final_tag);
                            }

                            found.push(DiscoveredModel {
                                name: model_name,
                                path: blob_path,
                                size_bytes: metadata.len(),
                                source: "Ollama".to_string(),
                            });
                        }
                    }
                }
            }
        }
    }

    found
}

/// Helper function to perform a unified search across all known and configured paths
pub fn find_all_local_models(configured_models_dir: &Path) -> Vec<DiscoveredModel> {
    let mut all_models = Vec::new();

    // 1. Scan user's explicitly configured directory
    let mut primary_models = scan_directory_for_gguf(configured_models_dir, "LocalCode Config");
    all_models.append(&mut primary_models);

    // 2. Scan Ollama
    let mut ollama_models = scan_ollama_cache();
    all_models.append(&mut ollama_models);

    // 3. Scan LM Studio default cache if it's not the primary dir already
    let lm_studio_dir = match shellexpand::tilde("~/.cache/lm-studio/models") {
        std::borrow::Cow::Borrowed(s) => PathBuf::from(s),
        std::borrow::Cow::Owned(s) => PathBuf::from(s),
    };

    if lm_studio_dir != configured_models_dir {
        let mut lm_models = scan_directory_for_gguf(&lm_studio_dir, "LM Studio");
        all_models.append(&mut lm_models);
    }

    all_models
}
