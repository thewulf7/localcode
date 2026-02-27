mod config;
mod profiling;
mod runner;
mod ui;

use anyhow::Result;
use clap::{Parser, Subcommand, Args as ClapArgs};
use console::style;

#[derive(Parser, Debug)]
#[command(author, version, about = "LocalCode - OpenCode Local LLM Setup", long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Perform initial setup and model selection for the global environment
    Setup(SetupArgs),
    /// Initialize a project-scoped configuration in the current directory
    Init(SetupArgs),
    /// Start the background LLM server using saved configuration
    Start,
    /// Show the real-time loading status of the background model
    Status,
    /// Stop the background LLM server
    Stop,
}

#[derive(ClapArgs, Debug)]
pub struct SetupArgs {
    /// Skip interactive prompts and accept defaults/arguments
    #[arg(short, long, default_value_t = false)]
    pub yes: bool,

    /// Specify the models to use directly (e.g. llama3-8b-instruct)
    #[arg(short, long)]
    pub models: Option<Vec<String>>,

    /// Do not use Docker to run llama.cpp (assumes native installation)
    #[arg(long, default_value_t = false)]
    pub no_docker: bool,

    /// Specify the port for the LLM API to bind to
    #[arg(short, long, default_value_t = 8080)]
    pub port: u16,

    /// Specify models directory target explicitly
    #[arg(long)]
    pub models_dir: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Commands::Status => {
            runner::show_status().await?;
        }
        Commands::Stop => {
            runner::stop_server().await?;
        }
        Commands::Start => {
            let config = config::load_localcode_config().await?;
            if config.run_in_docker {
                let model_names = config.models.iter().map(|m| m.name.clone()).collect::<Vec<_>>().join(", ");
                println!("{} {} with llama-swap in Docker on port {}...", 
                    style("🐳 Starting").blue(), 
                    style(&model_names).magenta().bold(), 
                    style(config.port).yellow()
                );
                
                if !config.models_dir.exists() {
                    tokio::fs::create_dir_all(&config.models_dir).await.unwrap_or(());
                }

                if let Err(e) = runner::download_models(&config.models, &config.models_dir).await {
                    println!("\n{} {}", style("❌ Failed to download models:").red().bold(), e);
                    std::process::exit(1);
                }

                if let Err(e) = runner::start_llama_swap_docker(&config.models, &config.models_dir, config.port).await {
                    println!("\n{} {}", style("❌ Failed to start Docker container:").red().bold(), e);
                    std::process::exit(1);
                }
                println!("{} {}", style("➜").cyan(), style("The model server is starting in the background. \n  Run `localcode status` to view its loading progress!").white().bold());
                println!("  {}", style("Run `localcode stop` later when you want to shut down the server.").dim());
            } else {
                let model_names = config.models.iter().map(|m| m.name.clone()).collect::<Vec<_>>().join(", ");
                println!("{} {} natively... (Not implemented in zero-config)", 
                    style("🚀 Starting").blue(),
                    style(&model_names).magenta().bold()
                );
            }
        }
        Commands::Setup(setup_args) => {
            println!("\n{}\n", style("✨ Welcome to OpenCode Global Setup! ✨").cyan().bold());

            // 1. Profile Hardware
            println!("{}", style("🔍 Profiling hardware capabilities via llmfit...").dim());
            let profile = profiling::profile_hardware().await?;
            println!("{} {}GB VRAM, {}GB RAM", 
                style("✓ Hardware Profile Detected:").green().bold(), 
                style(profile.vram_gb).yellow(), 
                style(profile.ram_gb).yellow()
            );

            // 2. Determine Optimal Model
            let recommended_model = match profile.vram_gb {
                v if v >= 24.0 => "llama3-70b-instruct",
                v if v >= 16.0 => "mixtral-8x7b-instruct",
                v if v >= 8.0 => "llama3-8b-instruct",
                _ => "phi3-mini",
            };

            // 3. User Interaction
            println!();
            let user_config = ui::prompt_user(&setup_args, &profile, recommended_model)?;
            println!();

            // 4. Configure OpenCode
            let provider_url = format!("http://localhost:{}/v1", user_config.port);
            let first_model_name = user_config.models.first().map(|m| m.name.clone()).unwrap_or_else(|| "default".to_string());
            config::configure_opencode(&first_model_name, &provider_url, false).await?;

            // 5. Download default skills
            config::download_initial_skills(&user_config.selected_skills).await?;

            // 6. Save configuration to disk
            config::save_localcode_config(&user_config, false).await?;

            println!("\n{}", style("🎉 Setup Complete! Global configuration saved.").green().bold());
            println!("{} {}", style("➜").cyan(), style("Run `localcode start` to boot up the LLM server!").white().bold());
        }
        Commands::Init(setup_args) => {
            println!("\n{}\n", style("✨ Initializing OpenCode Project Configuration ✨").cyan().bold());

            let profile = profiling::profile_hardware().await?;
            let recommended_model = match profile.vram_gb {
                v if v >= 24.0 => "llama3-70b-instruct",
                v if v >= 16.0 => "mixtral-8x7b-instruct",
                v if v >= 8.0 => "llama3-8b-instruct",
                _ => "phi3-mini",
            };

            let user_config = ui::prompt_user(&setup_args, &profile, recommended_model)?;
            println!();

            let provider_url = format!("http://localhost:{}/v1", user_config.port);
            let first_model_name = user_config.models.first().map(|m| m.name.clone()).unwrap_or_else(|| "default".to_string());
            
            // Pass true to indicate project-scoped
            config::configure_opencode(&first_model_name, &provider_url, true).await?;
            config::save_localcode_config(&user_config, true).await?;

            println!("\n{}", style("🎉 Project Initialization Complete! Local project configuration saved.").green().bold());
            println!("{} {}", style("➜").cyan(), style("Run `localcode start` to boot up the scoped LLM server!").white().bold());
        }
    }

    Ok(())
}
