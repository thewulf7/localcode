use anyhow::Result;
use std::path::PathBuf;
use tokio::fs;

use crate::ui::InitConfig;
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "skills/"]
pub struct SkillsAssets;

pub fn get_available_skills() -> Vec<String> {
    let mut skills = std::collections::HashSet::new();
    for file in SkillsAssets::iter() {
        let path = file.as_ref();
        if let Some(slash_idx) = path.find('/') {
            skills.insert(path[..slash_idx].to_string());
        }
    }
    let mut skills_vec: Vec<String> = skills.into_iter().collect();
    skills_vec.sort();
    skills_vec
}

pub async fn configure_opencode(
    models: &[crate::ui::ModelSelection],
    provider_url: &str,
    is_project: bool,
) -> Result<()> {
    let target_dir = if is_project {
        PathBuf::from(".opencode")
    } else {
        let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home_dir.join(".opencode")
    };

    let config_path = target_dir.join("config.json");

    if !target_dir.exists() {
        fs::create_dir_all(&target_dir).await?;
    }

    let mut config: serde_json::Value = if config_path.exists() {
        let existing_content = fs::read_to_string(&config_path).await?;
        serde_json::from_str(&existing_content).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        println!("📥 Initializing official OpenCode config template...");
        let template_content = include_str!("../opencode.json");
        serde_json::from_str(template_content).unwrap_or_else(|_| serde_json::json!({}))
    };

    let mut standard_model_name = "default".to_string();
    let mut autocomplete_model_name = None;

    for m in models {
        if crate::runner::is_autocomplete_model(&m.name) {
            if autocomplete_model_name.is_none() {
                autocomplete_model_name = Some(m.name.clone());
            }
        } else if standard_model_name == "default" {
            standard_model_name = m.name.clone();
        }
    }

    if let Some(obj) = config.as_object_mut() {
        obj.insert(
            "llm".to_string(),
            serde_json::json!({
                "provider": "custom",
                "model": standard_model_name,
                "api_base": provider_url,
            }),
        );

        if let Some(auto_name) = autocomplete_model_name {
            obj.insert(
                "tabAutocompleteModel".to_string(),
                serde_json::json!({
                    "provider": "custom",
                    "model": auto_name,
                    "api_base": provider_url,
                }),
            );
        }
    }

    println!(
        "💾 Writing OpenCode configuration to: {}",
        config_path.display()
    );
    fs::write(config_path, serde_json::to_string_pretty(&config)?).await?;

    Ok(())
}

pub async fn save_localcode_config(config: &crate::ui::InitConfig, is_project: bool) -> Result<()> {
    let target_dir = if is_project {
        PathBuf::from(".")
    } else {
        let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home_dir.join(".config").join("localcode")
    };
    let config_path = target_dir.join("localcode.json");

    if !target_dir.exists() {
        fs::create_dir_all(&target_dir).await?;
    }

    fs::write(config_path, serde_json::to_string_pretty(config)?).await?;
    Ok(())
}

pub async fn load_localcode_config() -> Result<crate::ui::InitConfig> {
    let mut config_path = PathBuf::from("localcode.json");

    if !config_path.exists() {
        let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        config_path = home_dir
            .join(".config")
            .join("localcode")
            .join("localcode.json");
    }

    if !config_path.exists() {
        anyhow::bail!("Global configuration not found. Please run `localcode init` first.");
    }

    let config_content = fs::read_to_string(config_path).await?;
    let config: crate::ui::InitConfig = serde_json::from_str(&config_content)?;
    Ok(config)
}

pub async fn download_initial_skills(selected_skills: &[String]) -> Result<()> {
    if selected_skills.is_empty() {
        return Ok(());
    }

    let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let skills_dir = home_dir.join(".opencode").join("skills");

    if !skills_dir.exists() {
        fs::create_dir_all(&skills_dir).await?;
    }

    println!("🚀 Installing selected OpenCode skills...");

    for file in SkillsAssets::iter() {
        let path = file.as_ref();
        let skill_name = if let Some(idx) = path.find('/') {
            &path[..idx]
        } else {
            continue;
        };

        if selected_skills.iter().any(|s| s == skill_name)
            && let Some(embedded_file) = SkillsAssets::get(path)
        {
            let dest_path = skills_dir.join(path);

            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent).await?;
            }

            fs::write(&dest_path, embedded_file.data).await?;
        }
    }

    println!("✅ Skills installed.");
    println!("✅ Skills installed.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_skills_loader() {
        let skills = get_available_skills();
        // Just verify it doesn't crash and returns the context7 folder we know exists in our tree based on UI selection choices
        assert!(skills.contains(&"context7".to_string()));
    }
}
