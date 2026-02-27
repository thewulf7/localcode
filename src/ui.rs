use crate::profiling::HardwareProfile;
use anyhow::Result;
use inquire::Confirm;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelSelection {
    pub name: String,
    pub quant: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SetupConfig {
    pub models: Vec<ModelSelection>,
    pub run_in_docker: bool,
    pub selected_skills: Vec<String>,
    pub models_dir: std::path::PathBuf,
    pub port: u16,
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

const AVAILABLE_SKILLS: &[&str] = &["context7"];

pub fn prompt_user(
    args: &crate::SetupArgs,
    profile: &HardwareProfile,
    recommended_model: &str,
) -> Result<SetupConfig> {
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
            vec![ModelSelection {
                name: recommended_model.to_string(),
                quant: profile
                    .recommended_models
                    .first()
                    .map(|m| m.best_quant.clone()),
            }]
        };

        return Ok(SetupConfig {
            models,
            run_in_docker: !args.no_docker,
            selected_skills: AVAILABLE_SKILLS.iter().map(|s| s.to_string()).collect(),
            models_dir: args.models_dir.clone().unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join(".opencode")
                    .join("models")
            }),
            port: args.port,
        });
    }

    let default_choice = args
        .models
        .as_ref()
        .and_then(|m| m.first())
        .map(|s| s.as_str())
        .unwrap_or(recommended_model);

    let is_dynamic = !profile.recommended_models.is_empty();

    let all_options: Vec<String> = if is_dynamic {
        profile
            .recommended_models
            .iter()
            .map(|m| format!("{} (Score: {}, Quant: {})", m.name, m.score, m.best_quant))
            .collect()
    } else {
        AVAILABLE_MODELS.iter().map(|&s| s.to_string()).collect()
    };

    let mut default_indices = Vec::new();
    if let Some(idx) = all_options.iter().position(|x| x.contains(default_choice)) {
        default_indices.push(idx);
    }

    let selected_options = inquire::MultiSelect::new(
        "Which models would you like to install and use?",
        all_options,
    )
    .with_default(&default_indices)
    .with_help_message("Use Space to select/deselect, Enter to confirm. Type to filter.")
    .with_page_size(10)
    .prompt()?;

    if selected_options.is_empty() {
        anyhow::bail!("You must select at least one model.");
    }

    let mut selected_models = Vec::new();
    for opt in selected_options {
        let mut final_model = opt.clone();
        let mut final_quant = None;
        if is_dynamic {
            if let Some(idx) = opt.find(" (") {
                final_model = opt[..idx].to_string();
            }
            if let Some(model) = profile
                .recommended_models
                .iter()
                .find(|m| m.name == final_model)
            {
                final_quant = Some(model.best_quant.clone());
            }
        }
        selected_models.push(ModelSelection {
            name: final_model,
            quant: final_quant,
        });
    }

    let run_in_docker = Confirm::new("Do you want to run this using llama.cpp via Docker?")
        .with_default(!args.no_docker)
        .with_help_message("This will automatically download and start the model without installing extra dependencies natively.")
        .prompt()?;

    // Fetch embedded skills dynamically
    let mut available_skills = crate::config::get_available_skills();

    // Attempt to prioritize context7 at the top
    if let Some(idx) = available_skills.iter().position(|s| s == "context7") {
        let context7 = available_skills.remove(idx);
        available_skills.insert(0, context7);
    }

    let default_models_dir = args.models_dir.clone().unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".opencode")
            .join("models")
    });

    let models_dir_str = inquire::Text::new("Where would you like to save models locally?")
        .with_default(&default_models_dir.to_string_lossy())
        .prompt()?;
    let models_dir = std::path::PathBuf::from(models_dir_str);

    let default_indices = (0..available_skills.len()).collect::<Vec<_>>();
    let selected_skills = inquire::MultiSelect::new(
        "Select initial OpenCode skills to install:",
        available_skills.clone(),
    )
    .with_default(&default_indices)
    .with_help_message("Use Space to select/deselect, Enter to confirm.")
    .prompt()?;

    Ok(SetupConfig {
        models: selected_models,
        run_in_docker,
        selected_skills,
        models_dir,
        port: args.port,
    })
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
    fn test_setup_config_serialize() {
        let config = SetupConfig {
            models: vec![ModelSelection {
                name: "test".to_string(),
                quant: None,
            }],
            run_in_docker: true,
            selected_skills: vec!["context7".to_string()],
            models_dir: std::path::PathBuf::from("/tmp/models"),
            port: 8080,
        };
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(serialized.contains("run_in_docker"));
        assert!(serialized.contains(r#""port":8080"#));
        assert!(serialized.contains("context7"));
    }
}
