# LocalCode User Guide

This guide covers every command, configuration option, and operational detail for LocalCode v0.1.12.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands Reference](#commands-reference)
  - [localcode init](#localcode-init)
  - [localcode start](#localcode-start)
  - [localcode status](#localcode-status)
  - [localcode stop](#localcode-stop)
  - [localcode ls](#localcode-ls)
  - [localcode upgrade](#localcode-upgrade)
- [Configuration](#configuration)
  - [localcode.json Schema](#localcodejson-schema)
  - [Project vs Global Scope](#project-vs-global-scope)
  - [llama.cpp Server Arguments](#llamacpp-server-arguments)
  - [OpenCode Integration](#opencode-integration)
- [Hardware Profiling](#hardware-profiling)
- [Model Discovery](#model-discovery)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Linux / macOS

```bash
curl -sL https://appcabin.io/install.sh | sh
```

The installer:
1. Detects your OS and architecture (`x86_64-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`).
2. Downloads the latest release tarball from GitHub.
3. Extracts the `localcode` binary to `~/.local/bin/`.
4. Optionally adds `~/.local/bin` to your `PATH` in `~/.bashrc` or `~/.zshrc`.

### Windows (PowerShell)

```powershell
irm https://appcabin.io/install.ps1 | iex
```

The binary is placed in `$env:LOCALAPPDATA\localcode\`.

### From Source

Requires a Rust toolchain (1.85+):

```bash
cargo install --git https://github.com/thewulf7/localcode.git
```

---

## Quick Start

```bash
# 1. Interactive setup (profiles hardware, picks models, downloads weights)
localcode init

# 2. Boot the Docker-based llama.cpp proxy
localcode start

# 3. Verify the server is running
localcode status

# 4. Point your IDE / OpenCode at http://localhost:8080/v1

# 5. When done
localcode stop
```

For a fully non-interactive setup:

```bash
localcode init --yes --global
localcode start
```

---

## Commands Reference

### `localcode init`

Initialize (or reinitialize) LocalCode configuration. This is the main setup command.

```
localcode init [OPTIONS]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--yes` | `-y` | `false` | Skip all interactive prompts, accept defaults or provided arguments |
| `--global` | | `false` | Save configuration globally (`~/.config/localcode/`) instead of the current directory |
| `--models <NAME>` | `-m` | auto | Specify model name(s) directly. Can be repeated: `-m model1 -m model2` |
| `--no-docker` | | `false` | Don't use Docker to run llama.cpp (assumes native installation) |
| `--port <PORT>` | `-p` | `8080` | Port for the LLM API to bind to |
| `--models-dir <PATH>` | | `~/.opencode/models` | Directory where GGUF weights are stored |

#### What happens during `init`

1. **Hardware Profiling** — `llmfit-core` detects VRAM, RAM, and computes compatible models/combos.
2. **Scope Selection** — Local (current directory) or Global (`~/.config/localcode/`).
3. **Model Selection** — Interactive picker showing hardware-scored options, filtered for coding models. In combo mode, a standard reasoning model is paired with a lightweight autocomplete model.
4. **Docker Preference** — Whether to run via Docker + llama-swap proxy.
5. **Models Directory** — Where to save downloaded GGUF weights.
6. **Skills Selection** — Optional OpenCode skill packs (e.g., `context7`).
7. **Configuration Saved** — Writes `localcode.json` and `.opencode/config.json`.

#### Examples

```bash
# Interactive setup (default — local scope)
localcode init

# Non-interactive global setup with specific models
localcode init --yes --global -m "qwen2-7b-instruct"

# Custom models directory and port
localcode init --models-dir ~/my-models --port 9090

# CPU-only setup (no Docker GPU passthrough)
localcode init --no-docker
```

---

### `localcode start`

Start the background LLM server using the saved configuration.

```
localcode start
```

This command:
1. Loads `localcode.json` (local first, then global fallback).
2. Ensures the models directory exists.
3. Downloads any missing GGUF weights from Hugging Face Hub.
4. Generates a `llama-swap` configuration mapping each model to a backend.
5. Launches the `ghcr.io/mostlygeek/llama-swap:latest` Docker container with:
   - GPU passthrough (`--gpus all`, with CPU fallback on failure).
   - Volume mounts for models and config.
   - Port binding to the configured port (default `8080`).

After starting, use `localcode status` to monitor model loading progress.

---

### `localcode status`

Show real-time loading status of the background model server.

```
localcode status
```

Queries the Docker container named `localcode-llm` and displays its current state.

---

### `localcode stop`

Stop the background LLM server gracefully.

```
localcode stop
```

Stops and removes the `localcode-llm` Docker container.

---

### `localcode ls`

List all locally available GGUF models across known cache locations.

```
localcode ls
```

Scans three sources:

| Source | Default Path | Method |
|--------|-------------|--------|
| **LocalCode Config** | Value of `models_dir` in `localcode.json` | Recursive `.gguf` file scan |
| **Ollama** | `~/.ollama/models/` | Parses manifest JSON → resolves blob digests |
| **LM Studio** | `~/.cache/lm-studio/models/` | Recursive `.gguf` file scan |

Output is a formatted table with **Name**, **Size**, and **Cache Source** columns.

```
✓ 4 Local Models Discovered
Name                                                         | Size         | Cache Source
-------------------------------------------------------------|--------------|---------------------
qwen2.5-coder-7b-instruct-Q4_K_M.gguf                       | 4.68 GB      | LocalCode Config
qwen2.5-coder-1.5b-instruct-Q8_0.gguf                       | 1.65 GB      | LocalCode Config
llama3:8b-latest                                             | 4.66 GB      | Ollama
codellama-7b-instruct.Q4_K_M.gguf                           | 3.80 GB      | LM Studio
```

---

### `localcode upgrade`

Self-update LocalCode to the latest GitHub release.

```
localcode upgrade
```

Downloads and replaces the current binary in-place. No package manager required.

---

## Configuration

### `localcode.json` Schema

```jsonc
{
  // Selected models with optional quantization override
  "models": [
    { "name": "qwen2.5-coder-7b-instruct", "quant": "Q4_K_M" },
    { "name": "qwen2.5-coder-1.5b-instruct", "quant": "Q8_0" }
  ],

  // Whether to use Docker-based llama.cpp + llama-swap
  "run_in_docker": true,

  // OpenCode skill packs to install during init
  "selected_skills": ["context7"],

  // Directory for downloaded GGUF weights (supports ~ expansion)
  "models_dir": "~/.opencode/models",

  // Port the LLM API binds to
  "port": 8080,

  // Optional llama.cpp server arguments (auto-populated by init)
  "llama_server_args": {
    "ctx_size": 8192,
    "n_gpu_layers": 999,
    "flash_attn": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "prompt-cache": "/models/prompt.cache",
    "prompt-cache-all": true
  }
}
```

### Project vs Global Scope

| Scope | Location | Created By |
|-------|----------|------------|
| **Local** (default) | `./localcode.json` in the working directory | `localcode init` |
| **Global** | `~/.config/localcode/localcode.json` | `localcode init --global` |

**Resolution order:** `localcode start` checks local first, then global. Local always wins.

### llama.cpp Server Arguments

The `llama_server_args` object is translated into CLI flags for the llama.cpp server process inside Docker:

| Key | CLI Flag | Description |
|-----|----------|-------------|
| `ctx_size` | `--ctx-size` | Maximum context length in tokens |
| `n_gpu_layers` | `--n-gpu-layers` | Number of layers to offload to GPU (`999` = all) |
| `flash_attn` | `--flash-attn` | Enable flash attention (GPU only) |
| `cache_type_k` | `--cache-type-k` | KV cache quantization for keys (`q8_0`, `f16`) |
| `cache_type_v` | `--cache-type-v` | KV cache quantization for values (`q8_0`, `f16`) |

**Any additional key-value pairs** in this object are passed through as `--key value` flags. Boolean `true` emits just the flag (`--mlock`); `false` is omitted. String/number values are appended as arguments.

Example with extra args:

```json
{
  "llama_server_args": {
    "ctx_size": 4096,
    "n_gpu_layers": 999,
    "flash_attn": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "numa": "numactl",
    "threads": 8,
    "mlock": true
  }
}
```

This produces:
```
--ctx-size 4096 --n-gpu-layers 999 --flash-attn --cache-type-k q8_0 --cache-type-v q8_0 --numa numactl --threads 8 --mlock
```

### OpenCode Integration

During `localcode init`, an `opencode/config.json` file is automatically generated (or updated) with the correct provider URL and model names:

- **Local scope**: `./.opencode/config.json`
- **Global scope**: `~/.opencode/config.json`

The config points the `llm` and `tabAutocompleteModel` providers at `http://localhost:<port>/v1`.

---

## Hardware Profiling

LocalCode uses [`llmfit-core`](https://crates.io/crates/llmfit-core) during `init` to detect:

- **VRAM** (GPU memory in GB)
- **RAM** (system memory in GB)

Based on these values, `llmfit-core` returns:

1. **Recommended Models** — Scored and ranked models that fit your hardware, with optimal quantization levels.
2. **Recommended Combos** — Pairs of a standard reasoning model + a small autocomplete model that fit simultaneously in VRAM.

The interactive picker prioritizes **coding-oriented** models (filtering by category). In `--yes` mode, the best coding combo is selected automatically.

### Auto-Configured llama.cpp Args

The `LlamaServerArgs` are computed from your hardware profile:

| VRAM | Context Size | GPU Layers | Flash Attention | KV Cache |
|------|-------------|------------|-----------------|----------|
| ≥ 24 GB | 32768 | 999 (all) | ✅ | q8_0 |
| ≥ 16 GB | 16384 | 999 (all) | ✅ | q8_0 |
| ≥ 12 GB | 8192 | 999 (all) | ✅ | q8_0 |
| ≥ 8 GB | 4096 | 999 (all) | ✅ | q8_0 |
| < 8 GB GPU | 2048 | 999 (all) | ✅ | q8_0 |
| CPU only (≥ 32 GB RAM) | 8192 | 0 | ❌ | f16 |
| CPU only (≥ 16 GB RAM) | 4096 | 0 | ❌ | f16 |
| CPU only (< 16 GB RAM) | 2048 | 0 | ❌ | f16 |

---

## Model Discovery

The `localcode ls` command finds GGUF models without downloading anything. It scans:

### 1. Configured Models Directory
A recursive walk of your `models_dir` looking for any file ending in `.gguf`.

### 2. Ollama Cache
Reads Ollama's manifest structure at `~/.ollama/models/manifests/registry.ollama.ai/`. For each manifest JSON, it:
- Finds layers with `mediaType: "application/vnd.ollama.image.model"`.
- Resolves the `digest` to a blob file in `~/.ollama/models/blobs/`.
- Extracts a human-readable name from the directory hierarchy (e.g., `llama3:8b-latest`).

### 3. LM Studio Cache
A recursive walk of `~/.cache/lm-studio/models/` for `.gguf` files, unless this path is the same as your configured `models_dir`.

---

## Troubleshooting

### NVIDIA Container Toolkit Not Detected

**Symptom:** `localcode start` crashes when Docker tries `--gpus all`.

**Fix:** Install the NVIDIA Container Toolkit for your platform. On Windows, ensure WSL2 GPU passthrough is configured. LocalCode automatically falls back to CPU mode (`--gpus 0`) if GPU initialization fails.

### Download Times Out

**Symptom:** `localcode init` or `localcode start` hangs during model download.

**Fix:**
- Check that your network allows HTTPS connections to `huggingface.co`.
- Disable any SSL inspection proxies.
- Press `Ctrl+C` to abort — the downloader supports resume, so rerunning will continue from where it stopped.

### `Global configuration not found`

**Symptom:** `localcode start` says "Please run `localcode init` first."

**Fix:** Run `localcode init` (or `localcode init --global`) to create the configuration. `localcode start` looks for `./localcode.json` first, then `~/.config/localcode/localcode.json`.

### Docker Container Already Exists

**Symptom:** `localcode start` fails because `localcode-llm` container already exists.

**Fix:** Run `localcode stop` first, then `localcode start` again.

### Models Not Showing in `localcode ls`

**Symptom:** You know you have models downloaded but `ls` shows nothing.

**Fix:** Ensure the files have the `.gguf` extension. Ollama uses blob files (no extension) — these are detected via manifest parsing, not file extension. If Ollama models aren't appearing, check that `~/.ollama/models/manifests/` exists and contains valid JSON manifests.
