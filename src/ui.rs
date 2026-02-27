use anyhow::Result;
use inquire::{Confirm, Select};
use crate::profiling::HardwareProfile;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct SetupConfig {
    pub model_name: String,
    pub quant: Option<String>,
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

const AVAILABLE_SKILLS: &[&str] = &[
    "context7",
];

pub fn prompt_user(args: &crate::SetupArgs, profile: &HardwareProfile, recommended_model: &str) -> Result<SetupConfig> {
    if args.yes {
        return Ok(SetupConfig {
            model_name: args.model.clone().unwrap_or_else(|| recommended_model.to_string()),
            quant: profile.recommended_models.first().map(|m| m.best_quant.clone()),
            run_in_docker: !args.no_docker,
            selected_skills: AVAILABLE_SKILLS.iter().map(|s| s.to_string()).collect(),
            models_dir: args.models_dir.clone().unwrap_or_else(|| {
                dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join(".opencode").join("models")
            }),
            port: args.port,
        });
    }

    let default_choice = args.model.as_deref().unwrap_or(recommended_model);
    
    let mut top_options = Vec::new();
    let is_dynamic = !profile.recommended_models.is_empty();
    
    if is_dynamic {
        for m in profile.recommended_models.iter().take(5) {
            top_options.push(format!("{} (Score: {}, Quant: {})", m.name, m.score, m.best_quant));
        }
    } else {
        top_options = AVAILABLE_MODELS.iter().take(5).map(|s| s.to_string()).collect();
        if !top_options.contains(&default_choice.to_string()) {
            if top_options.len() >= 5 {
                top_options.pop();
            }
            top_options.insert(0, default_choice.to_string());
        }
    }
    
    let view_all_option = "View all models...".to_string();
    top_options.push(view_all_option.clone());
    
    // Attempt to guess cursor if we match a substring
    let starting_cursor = top_options.iter().position(|x| x.contains(default_choice)).unwrap_or(0);

    let mut selected_option = Select::new("Which model would you like to use?", top_options)
        .with_starting_cursor(starting_cursor)
        .with_help_message("Use up/down arrows to scroll. Type to filter options.")
        .prompt()?;

    if selected_option == view_all_option {
        if is_dynamic {
            let all_options: Vec<String> = profile.recommended_models.iter()
                .map(|m| format!("{} (Score: {}, Quant: {})", m.name, m.score, m.best_quant)).collect();
            selected_option = Select::new("Select from all recommended models:", all_options)
                .with_page_size(10)
                .prompt()?;
        } else {
            let all_options: Vec<String> = AVAILABLE_MODELS.iter().map(|s| s.to_string()).collect();
            selected_option = Select::new("Select from all available models:", all_options)
                .with_page_size(10)
                .prompt()?;
        }
    }
    
    // Parse out real model name and quant
    let mut final_model = selected_option.clone();
    let mut final_quant = None;
    if is_dynamic {
        // extract string before the first ' ('
        if let Some(idx) = selected_option.find(" (") {
            final_model = selected_option[..idx].to_string();
        }
        if let Some(model) = profile.recommended_models.iter().find(|m| m.name == final_model) {
            final_quant = Some(model.best_quant.clone());
        }
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
        dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".opencode").join("models")
    });

    let models_dir_str = inquire::Text::new("Where would you like to save models locally?")
        .with_default(&default_models_dir.to_string_lossy())
        .prompt()?;
    let models_dir = std::path::PathBuf::from(models_dir_str);

    let default_indices = (0..available_skills.len()).collect::<Vec<_>>();
    let selected_skills = inquire::MultiSelect::new("Select initial OpenCode skills to install:", available_skills.clone())
        .with_default(&default_indices)
        .with_help_message("Use Space to select/deselect, Enter to confirm.")
        .prompt()?;

    Ok(SetupConfig { 
        model_name: final_model, 
        quant: final_quant,
        run_in_docker,
        selected_skills,
        models_dir,
        port: args.port,
    })
}
