use anyhow::Result;
use std::path::PathBuf;
use tokio::fs;

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
        // Build the models map with both standard and autocomplete entries
        let mut models_map = serde_json::Map::new();
        models_map.insert(
            standard_model_name.clone(),
            serde_json::json!({
                "name": standard_model_name,
            }),
        );
        if let Some(ref auto_name) = autocomplete_model_name {
            models_map.insert(
                auto_name.clone(),
                serde_json::json!({
                    "name": auto_name,
                }),
            );
        }

        // Build the provider config with model + small_model top-level keys
        let mut provider_obj = serde_json::Map::new();
        provider_obj.insert("models".to_string(), serde_json::Value::Object(models_map));
        provider_obj.insert("model".to_string(), serde_json::json!(standard_model_name));
        if let Some(ref auto_name) = autocomplete_model_name {
            provider_obj.insert("small_model".to_string(), serde_json::json!(auto_name));
        }
        provider_obj.insert("name".to_string(), serde_json::json!("LocalCode"));
        provider_obj.insert("npm".to_string(), serde_json::json!("@ai-sdk/openai-compatible"));
        provider_obj.insert(
            "options".to_string(),
            serde_json::json!({
                "provider": "openai",
                "baseURL": provider_url,
            }),
        );

        // Nest under provider.localcode
        let provider = serde_json::json!({ "localcode": serde_json::Value::Object(provider_obj) });
        obj.insert("provider".to_string(), provider);

        // Remove legacy keys if present from older configs
        obj.remove("llm");
        obj.remove("tabAutocompleteModel");
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
