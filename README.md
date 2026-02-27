# LocalCode

LocalCode is a streamlined, command-line utility for dynamically launching and swapping open weights large language models (LLMs) via `llama-swap`. It acts as the backbone for OpenCode by abstracting away the hassle of managing container instances, finding specific weights, and setting correct proxy pointers for local AI capabilities.

## Key Features

- **Hardware Profiling:** Automatically detects your available system VRAM and RAM using `llmfit` to recommend models capable of running on your machine natively.
- **Dynamic Swapping:** Utilizes `llama-swap` as a robust reverse proxy. You configure what `.gguf` files you want and dynamically switch models purely by requesting them. 
- **Parallel Autocompletion Constraints:** Intelligent group detection ensures small, parameter-efficient completion models remain loaded parallel to active chat contexts, meaning zero load latency when asking your IDE for completions while chatting!
- **Project-Level & Global Configuration:** Stores state either globally in `~/.config/localcode/localcode.json` or explicitly overrides properties by using `localcode init` so any models spawned inside a project have domain specific characteristics.

---

## Tech Stack

- **Language:** Rust
- **Distribution Ecosystem:** Cargo / crates.io dependencies
- **Inference Runtime Engine:** ggml-org/llama.cpp (via wrapper)
- **Local Proxy Engine:** ghcr.io/mostlygeek/llama-swap 
- **Model Downloads:** `hf-hub` native sync downloading interface

---

## Prerequisites

- Standard `cargo` installation (part of the standard rustup configuration)
- Docker Desktop or Podman installed on the OS supporting local volume mounting capability
- Minimum 8 GB RAM (though 16 GB+ is recommended for optimal inference memory scaling)
- If taking advantage of Nvidia Acceleration, you MUST install the NVIDIA Container Toolkit or enable WSL2 passthrough cleanly.

---

## Getting Started

### 1. Installation

To configure and run localcode, simply compile the bin and execute the initial setup command anywhere.

```bash
cargo build --release
cd target/release/
./localcode setup
```

This guides you through hardware detection and model download choices.
If you simply want to accept defaults without interactive prompts:
```bash
./localcode setup --yes -m "llama3-8b-instruct" -m "qwen2.5-coder-1.5b-instruct"
```

### 2. Starting the Environment 

```bash
# Deploys Llama-Swap pulling your chosen parameters cleanly!
./localcode start
```

### 3. Monitoring Operation Logging

Once running as a background service container, monitor the active model being mapped to port memory in real time easily:

```bash
# Attach terminal std/out directly
./localcode status

# To kill it safely when done:
./localcode stop
```

---

## Configuration

If you'd like to adjust specific overrides or apply models on a per-project boundary, simply navigate inside your relevant project via the CLI.

```bash
# This forces the generator to initialize a ./localcode.json properties mapping.
./localcode init
```

The system will subsequently honor the local config overrides whenever you run `localcode start` from that directory!

## Troubleshooting

### Q: GPU Fallback Prompt Displays `NVIDIA Container Toolkit not detected` on Docker Startup
A: If Docker attempts to access GPU via `--gpus all` and it crashes, our runtime executor catches this anomaly safely. Ensure you've setup Windows WSL mapped drivers properly, otherwise LocalCode gracefully injects `--gpus 0` allowing basic operation CPU fallback processing!

### Q: Download times out consistently on `cargo code start`
A: The internal handler securely verifies HF parameters to download the matching file. Ensure no system firewall interferes with your Hugging Face HTTPS connection hooks!
