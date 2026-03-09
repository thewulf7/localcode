<div align="center">
  <img src="https://raw.githubusercontent.com/thewulf7/localcode/master/assets/logo.png" alt="LocalCode Logo" width="200"/>
  <h1>LocalCode</h1>
  <p><strong>A streamlined, developer-first command-line utility for dynamically launching and swapping open-weights Large Language Models (LLMs).</strong></p>

  <p>LocalCode acts as the invisible intelligence backbone, abstracting away container management, hardware-based model selection, and memory constraints so you can focus on building.</p>

  <p>
    <a href="https://github.com/thewulf7/localcode/actions"><img src="https://img.shields.io/github/actions/workflow/status/thewulf7/localcode/build.yml?branch=master" alt="Build Status"></a>
    <a href="https://crates.io/crates/localcode"><img src="https://img.shields.io/crates/v/localcode.svg" alt="Crates.io"></a>
    <a href="https://github.com/thewulf7/localcode/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  </p>
</div>

---

## 🚀 Key Features

*   **Intelligent Hardware Profiling:** Uses `llmfit-core` to auto-detect VRAM, RAM, GPU backend (CUDA/Metal/Vulkan/ROCm), CPU cores, and unified memory. Every llama.cpp parameter — context size, GPU layers, threads, KV cache quantization, flash attention, parallel slots — is calculated automatically. No manual tuning needed.
*   **VRAM-Aware Context Sizing:** Context size is computed from a dual-constraint formula: VRAM budget (what physically fits) capped by quality ceiling (native context × YaRN extension factor). See GUIDE.md for the math.
*   **Dual Model Combos:** Automatically suggests ideal combinations of large reasoning models alongside lightning-fast **autocomplete models** based on your available VRAM footprint.
*   **Claude Code + OpenCode Support:** Works with both OpenCode (OpenAI-compatible API) and Claude Code (Anthropic Messages API) on the same port. Model aliases for all Claude 3.5/4.x IDs are configured automatically.
*   **Dynamic Swapping:** Seamlessly switches models in and out of memory at the proxy layer using `llama-swap`. Request a different model and watch it swap instantly.
*   **Zero Port Conflicts:** Both your heavy chat model and your instantaneous autocomplete model run on the *exact same port* (`8080`). The proxy handles the routing natively.
*   **Model Discovery:** Automatically scans known cache locations — **Ollama**, **LM Studio**, and any custom directory — so you can reuse weights you already have on disk (`localcode ls`).
*   **Configurable llama.cpp Args:** Fine-tune all parameters in `localcode.json` or let `init` auto-configure everything from hardware profiling.
*   **Global & Project Contexts:** Store state globally or override per-project (`localcode init`).
*   **One-Line Installation:** Install immediately with secure OS-specific scripts without requiring a Rust toolchain.
*   **Self-Upgrade:** Update the binary in-place from GitHub releases (`localcode upgrade`).

---

## 🛠️ Prerequisites

LocalCode runs via containerization. You must have:

*   **Docker Desktop** or **Podman** installed on the host OS.
*   If utilizing NVIDIA acceleration on Windows, you must configure **WSL2 passthrough** and have the **NVIDIA Container Toolkit** correctly mapped.
*   Minimum **8 GB RAM** (16 GB+ is strongly recommended for practical inference scaling).

---

## ⚡ Quick Start

### 1. Installation

**For Linux / macOS:**
```bash
curl -sL https://appcabin.io/install.sh | sh
```

**For Windows (PowerShell):**
```powershell
irm https://appcabin.io/install.ps1 | iex
```

*<small>Alternatively, if you have a Rust toolchain installed, you can build from source: `cargo install --git https://github.com/thewulf7/localcode.git`</small>*

### 2. Initial Setup

Configure your models and directories for the first time. LocalCode will profile your hardware, recommend model/quantization combos, and download them automatically from Hugging Face:

```bash
localcode init
```

During setup you will be prompted to choose:
- **Scope** — Local (current project) or Global (`~/.config/localcode/`).
- **Model Mode** — Single model or a Normal + Autocomplete combo.
- **Models** — Picked from hardware-profiled recommendations filtered for coding tasks.
- **Docker** — Whether to use the Docker-based llama.cpp backend.
- **Models directory** — Where to store downloaded GGUF weights.

**Headless Setup (CI/CD / Automation):**
```bash
localcode init --yes --global -m "llama3-8b-instruct" -m "qwen2.5-coder-1.5b-instruct"
```

### 3. Start the Server

Deploys the reverse proxy mapping across Docker and orchestrates the weights:

```bash
localcode start
```

### 4. Status & Shutdown

```bash
# Check the container lifecycle and proxy mapping
localcode status

# Stop background services
localcode stop
```

### 5. Discover Local Models

List all `.gguf` weights already present on your system (Ollama, LM Studio, or any configured directory):

```bash
localcode ls
```

### 6. Self-Update

```bash
localcode upgrade
```

> 📖 **For a complete command reference, configuration guide, and troubleshooting docs see [GUIDE.md](GUIDE.md).**

---

## 🏛️ Architecture

LocalCode is composed of highly predictable, independent components communicating via a local service mesh structure.

### Directory Structure & Config Map

The system uses `~/.config/localcode/` (or OS equivalent) to maintain its global definition map:

```
~/.config/localcode/
 ├── localcode.json       # Central Configuration State
 └── models/              # Downloaded HuggingFace GGUF Weights (default)
```

*Note: You can override your model path explicitly using `localcode init --models-dir /my/custom/path`.*

### Request Lifecycle Flow

When an inference call is made from OpenCode, Claude Code, or any other frontend, LocalCode abstracts the execution:

```mermaid
graph LR
    A[OpenCode / IDE Plugin] -->|OpenAI API /v1/chat/completions| B(llama-swap Proxy :8080)
    A2[Claude Code] -->|Anthropic API /v1/messages| B
    B -->|Route by model alias & load GGUF| C[llama.cpp Backend]
    C -->|GGUF Binary Mapping| D[(Model Disk / ~/.config/localcode/models/)]
    C -->|Execute Inference| B
    B -->|Return JSON| A
    B -->|Return JSON| A2
```

1. **Proxy Intercept:** The llama-swap reverse proxy listens on the configured port (default `8080`).
2. **Context Resolution:** The proxy inspects the model ID in the request. Claude Code sends `claude-sonnet-4-6` etc. — these are aliased to your local model.
3. **Weight Loading:** If the requested model is not active, the proxy unloads the current model and loads the new one from disk.
4. **Inference:** The request is forwarded to llama.cpp with grammar-constrained tool-call generation (`tool_choice: any`), YaRN rope scaling, and the model's native Jinja chat template.

> The Docker image `ghcr.io/thewulf7/localcode:cuda-latest` bundles llama-swap + llama-server with CUDA 12.8 support.

### Model Discovery

`localcode ls` scans three sources to find existing `.gguf` weights:

| Source | Path |
|--------|------|
| **LocalCode Config** | The directory specified in `localcode.json` → `models_dir` |
| **Ollama** | `~/.ollama/models/blobs/` (parsed from manifests) |
| **LM Studio** | `~/.cache/lm-studio/models/` |

---

## 📝 Configuration

### Project-Level Overrides

Need specific models configured just for one project? `localcode init` defaults to local scope. To explicitly use global scope instead:

```bash
localcode init --global
```

A local `./localcode.json` in the working directory always takes precedence over the global configuration.

### llama.cpp Server Arguments

The `llama_server_args` key in `localcode.json` controls how the llama.cpp backend is launched. These are auto-populated based on your hardware during `init`, but you can manually tune them:

```json
{
  "llama_server_args": {
    "ctx_size": 49152,
    "n_gpu_layers": 999,
    "flash_attn": "on",
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "threads": 8,
    "parallel": 2,
    "mlock": true,
    "slot-save-path": "/models"
  }
}
```

All parameters are calculated automatically during `init` based on your VRAM, GPU backend, CPU cores, and model size. Any additional key-value pairs are passed through directly as `--key value` flags to llama.cpp.

---

## 🔌 Client Configuration

Once the server is running, you can connect your favorite AI-powered coding tools.

### 1. OpenCode

To use your local server in OpenCode, update your `opencode.json` (found in `~/.opencode/config.json` or your project's `.opencode/config.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "your-model-name",
  "small_model": "your-small-model-name",
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 3000
  },
  "provider": {
    "localcode": {
      "models": {
        "your-model-name": {
          "name": "your-model-name"
        },
        "your-small-model-name": {
          "name": "your-small-model-name"
        }
      },
      "name": "LocalCode",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "provider": "openai",
        "baseURL": "http://localhost:8080/v1"
      }
    }
  }
}
```

The `model` key sets the primary reasoning model and `small_model` sets the fast autocomplete model. Both run on the same port — the llama-swap proxy routes requests based on the model name.

### 2. Claude Code

LocalCode natively supports the Anthropic Messages API (`/v1/messages`). Claude Code connects to the same port as OpenCode — all Claude model IDs (3.5, 4.x series) are aliased to your local model automatically.

Set the following environment variables in your terminal:

**macOS / Linux:**
```bash
export ANTHROPIC_BASE_URL="http://localhost:8080"
export ANTHROPIC_API_KEY="sk-localcode"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=42132
claude
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_BASE_URL="http://localhost:8080"
$env:ANTHROPIC_API_KEY="sk-localcode"
$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS=42132
claude
```

> [!IMPORTANT]
> **`CLAUDE_CODE_MAX_CONTEXT_TOKENS` must be aligned with your model's `ctx_size`.** Claude Code uses this value to decide how much conversation history, system prompt, and tool definitions to pack into each request. If it exceeds the model's actual context window, you'll get a `400 exceed_context_size` error.
>
> **Formula:** `CLAUDE_CODE_MAX_CONTEXT_TOKENS = ctx_size - response_headroom`
>
> Reserve ~15% of ctx_size (minimum 4096 tokens) for the model's response. For example:
> | `ctx_size` | `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Response headroom |
> |------------|--------------------------------|-------------------|
> | 16384      | 12288                          | 4096              |
> | 32768      | 28672                          | 4096              |
> | 49152      | 42132                          | 7020              |
> | 65536      | 56196                          | 9340              |
>
> Run `localcode info` to see the exact values calculated for your configuration.

> [!TIP]
> Run `localcode info` anytime to see your current configuration and copy-paste these commands! The proxy also handles:
> - **Tool call generation** — grammar-constrained via `tool_choice: { type: "any" }`
> - **Sampling parameter isolation** — strips Claude Code's cloud-tuned `temperature`/`top_k`/`top_p` to preserve local model quality
> - **YaRN context extension** — extends context beyond the model's native training length via `--rope-scaling yarn`


---

## 🧠 Advanced: Custom Skills (Recommended)

To provide your local models with better tool-use capabilities and project awareness, we recommend adding specific skills to your client.

### Manual Installation for OpenCode

Since large skill banks can sometimes exceed local context windows, we recommend manually copying specific skills into your `.opencode` directory:

1. Create a `skills` folder if it doesn't exist: `mkdir .opencode/skills`
2. Download or copy your desired `.md` or `.json` skills into that folder.
3. Restart your OpenCode session.

> [!TIP]
> Use the **context7** skill to provide high-fidelity project navigation and structure awareness to your local model. You can find reference skills in the `skills/` directory of this repository.

---

## 🆘 Troubleshooting

### `NVIDIA Container Toolkit not detected`
**Symptom:** During `localcode start`, Docker attempts to access the GPU (`--gpus all`) and the initialization crashes.
**Solution:** Ensure you've cleanly installed runtime configurations for Windows WSL mapped drivers. If GPU allocation is irreversibly misconfigured, LocalCode acts gracefully by catching the Docker API bounds error and injecting `--gpus 0`, enabling immediate **CPU fallback processing**.

### Download Times Out Setting Up Models
**Symptom:** `localcode init` halts indefinitely while fetching GGUF weights.
**Solution:** The internal handler syncs securely with Hugging Face Hub limits. Ensure your network doesn't possess SSL inspection hooks obstructing standard HTTPS payload transfers. You can safely abort (`Ctrl+C`) and retry `localcode init`, and the internal downloader will gracefully resume the cached blob segments.

### `Global configuration not found`
**Symptom:** `localcode start` fails with "Please run `localcode init` first."
**Solution:** Run `localcode init --global` to create the system-wide configuration, or ensure a local `localcode.json` exists in your working directory.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check [issues page](https://github.com/thewulf7/localcode/issues).

When submitting PRs, ensure you adhere to the project's formatting by executing:
```sh
cargo clippy -- -D warnings
cargo fmt --check
cargo test
```
