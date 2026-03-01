mod config;
mod profiling;
mod runner;
mod ui;

use anyhow::Result;
use clap::{Args as ClapArgs, Parser, Subcommand};
use console::style;
use self_update::cargo_crate_version;

#[derive(Parser, Debug)]
#[command(author, version, about = "LocalCode - OpenCode Local LLM Setup", long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Initialize configuration for LocalCode (defaults to local project, use --global for system-wide)
    Init(InitArgs),
    /// Starts the localcode update engine to fetch the newest github release
    Upgrade,
    /// Start the background LLM server using saved configuration
    Start,
    /// Show the real-time loading status of the background model
    Status,
    /// Stop the background LLM server
    Stop,
}

#[derive(ClapArgs, Debug)]
pub struct InitArgs {
    /// Skip interactive prompts and accept defaults/arguments
    #[arg(short, long, default_value_t = false)]
    pub yes: bool,

    /// Set configuration globally in ~/.config/localcode instead of current directory
    #[arg(long, default_value_t = false)]
    pub global: bool,

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
                let model_names = config
                    .models
                    .iter()
                    .map(|m| m.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "{} {} with llama-swap in Docker on port {}...",
                    style("🐳 Starting").blue(),
                    style(&model_names).magenta().bold(),
                    style(config.port).yellow()
                );

                if !config.models_dir.exists() {
                    tokio::fs::create_dir_all(&config.models_dir)
                        .await
                        .unwrap_or(());
                }

                if let Err(e) = runner::download_models(&config.models, &config.models_dir).await {
                    println!(
                        "\n{} {}",
                        style("❌ Failed to download models:").red().bold(),
                        e
                    );
                    std::process::exit(1);
                }

                if let Err(e) =
                    runner::start_llama_swap_docker(&config.models, &config.models_dir, config.port)
                        .await
                {
                    println!(
                        "\n{} {}",
                        style("❌ Failed to start Docker container:").red().bold(),
                        e
                    );
                    std::process::exit(1);
                }
                println!("{} {}", style("➜").cyan(), style("The model server is starting in the background. \n  Run `localcode status` to view its loading progress!").white().bold());
                println!(
                    "  {}",
                    style("Run `localcode stop` later when you want to shut down the server.")
                        .dim()
                );
            } else {
                let model_names = config
                    .models
                    .iter()
                    .map(|m| m.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "{} {} natively... (Not implemented in zero-config)",
                    style("🚀 Starting").blue(),
                    style(&model_names).magenta().bold()
                );
            }
        }
        Commands::Upgrade => {
            println!("{}", style("Checking for updates...").dim());
            
            let status = self_update::backends::github::Update::configure()
                .repo_owner("thewulf7")
                .repo_name("localcode")
                .bin_name("localcode")
                .show_download_progress(true)
                .current_version(cargo_crate_version!())
                .build()
                .unwrap()
                .update()?;

            println!(
                "{} {}",
                style("Update status:").green().bold(),
                status.version()
            );
        }
        Commands::Init(init_args) => {
            println!(
                "\n{}\n",
                style("✨ Initialize LocalCode Environment ✨")
                    .cyan()
                    .bold()
            );

            // 1. Profile Hardware
            println!(
                "{}",
                style("🔍 Profiling hardware capabilities via llmfit...").dim()
            );
            let profile = profiling::profile_hardware().await?;
            println!(
                "{} {}GB VRAM, {}GB RAM",
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
            let (user_config, is_project_scoped) = ui::prompt_user(&init_args, &profile, recommended_model)?;
            println!();

            // 4. Configure OpenCode
            let provider_url = format!("http://localhost:{}/v1", user_config.port);
            config::configure_opencode(&user_config.models, &provider_url, is_project_scoped).await?;

            // 5. Download default skills
            if !is_project_scoped {
                config::download_initial_skills(&user_config.selected_skills).await?;
            }

            // 6. Save configuration to disk
            config::save_localcode_config(&user_config, is_project_scoped).await?;

            let scope_str = if is_project_scoped { "Local project" } else { "Global system" };
            println!(
                "\n{}",
                style(format!("🎉 Initialization Complete! {} configuration saved.", scope_str))
                    .green()
                    .bold()
            );
            println!(
                "{} {}",
                style("➜").cyan(),
                style("Run `localcode start` to boot up the LLM server!")
                    .white()
                    .bold()
            );
        }
    }

    Ok(())
}
