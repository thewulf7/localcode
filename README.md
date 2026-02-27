# LocalCode

A standalone, blazing-fast CLI tool built in Rust to instantly give you a seamless "it just works" local LLM environment via Docker & `llama.cpp`. Stop fighting endless configuration scripts and Python dependencies, and get back to writing code.

LocalCode checks your hardware profile, downloads optimal models by default, spins up a Docker container on-demand, and transparently connects OpenCode directly to it.

## Key Features

- **Zero-Config Setup**: One terminal command, and you're chatting with a local AI.
- **Dependency-Free**: A single, pre-compiled Rust binary. No Node modules, no pip environments.
- **Intelligent Hardware Profiling**: Automatically detects available VRAM and defaults to an appropriately-sized model (falling back all the way to `Phi-3-mini` if you have sparse resources).
- **Interactive TUI**: Uses styling inspired by Claude Code to let you seamlessly confirm or override the setup.
- **Docker-native `llama.cpp`**: Skips the pain of building `llama.cpp` natively by talking directly to the `ggml-org/llama.cpp` Docker image.

## Tech Stack

- **Language**: Rust
- **CLI Framework**: Clap
- **Interactive UI**: inquire
- **Inference Ecosystem**: Docker + `llama.cpp`

## Prerequisites

- **Docker Desktop / Daemon** (must be installed and running)
- An operating system capable of running Rust binaries.

*(Note: for GPU inference, ensure the Nvidia Container Toolkit is set up for Docker, otherwise it will run on CPU).*

## Getting Started

### 1. Build from Source

```bash
git clone https://github.com/your-username/localcode.git
cd localcode
cargo build --release
```

### 2. Run LocalCode

LocalCode includes an interactive prompt that will guide you through the process, confirm your hardware choices, and initialize the setup.

```bash
./target/release/localcode
```

### 3. CLI Arguments

If you wish to bypass the interactive UI or configure advanced settings like server port or custom models, you can use the built-in CLI flags:

| Flag | Description | Default |
| --- | --- | --- |
| `-y, --yes` | Skip interactive prompts and accept all defaults | `false` |
| `-m, --model <MODEL>` | Specify the model identifier directly (e.g. `phi3-mini`) | `None` (auto-detected) |
| `--no-docker` | Do not use Docker. Assumes `llama.cpp` is natively installed. | `false` |
| `-p, --port <PORT>` | the port for the LLM API to bind to | `8080` |

#### Examples

**Start the recommended model with no prompts:**
```bash
localcode -y
```

**Boot a lightweight model on port 9000:**
```bash
localcode -y -m "phi3-mini" -p 9000
```

## Architecture

The project contains the following flow:
1. `src/main.rs`: Entrypoint. Parses options via `clap`.
2. `src/profiling.rs`: Resolves mocked or real system hardware specs to identify VRAM thresholds.
3. `src/ui.rs`: Drives the `inquire` interactive prompt, skipping if `--yes` was provided.
4. `src/runner.rs`: Dispatches the Docker run command using the `--hf-file` mechanism of `llama.cpp`, mapping volumes and exposing ports automatically.
5. `src/config.rs`: Modifies `~/.opencode/config.json` specifically to integrate the new server interface.

## Troubleshooting

### Error: `Docker is not running or not installed correctly`
Make sure you have Docker correctly installed from [Docker's official site](https://docs.docker.com/get-docker/). If you are using Linux, verify that your user is added to the `docker` group or that the daemon is correctly active.

### Container starts but inference is dreadfully slow
Docker will default to CPU inference if the `--gpus all` mapping fails to attach your hardware. Ensure you have installed the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and restarted the docker service.
