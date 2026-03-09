# LocalCode User Guide

This guide covers every command, configuration option, and operational detail for LocalCode v0.1.33.

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
  - [localcode info](#localcode-info)
- [Configuration](#configuration)
  - [localcode.json Schema](#localcodejson-schema)
  - [Project vs Global Scope](#project-vs-global-scope)
  - [llama.cpp Server Arguments](#llamacpp-server-arguments)
  - [OpenCode Integration](#opencode-integration)
  - [Claude Code Integration](#claude-code-integration)
- [Hardware Profiling](#hardware-profiling)
  - [Auto-Configured llama.cpp Args](#auto-configured-llamacpp-args)
  - [Native Context Lengths](#native-context-lengths)
- [llama-swap Proxy Layer](#llama-swap-proxy-layer)
- [Model Discovery](#model-discovery)
- [Context Token Alignment (Claude Code)](#context-token-alignment-claude-code)
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

1. **Hardware Profiling** — `llmfit-core` detects VRAM, RAM, GPU backend, CPU cores, unified memory, and computes compatible models/combos.
2. **Scope Selection** — Local (current directory) or Global (`~/.config/localcode/`).
3. **Model Selection** — Interactive picker showing hardware-scored options, filtered for coding models. In combo mode, a standard reasoning model is paired with a lightweight autocomplete model.
4. **Docker Preference** — Whether to run via Docker + llama-swap proxy.
5. **Models Directory** — Where to save downloaded GGUF weights.
6. **Auto-Configuration** — Calculates all `llama_server_args` from your hardware profile (ctx_size, GPU layers, threads, parallel, flash_attn, KV cache quant, mlock).
7. **Configuration Saved** — Writes `localcode.json` and `.opencode/config.json`.
8. **Config Instructions** — Displays the `localcode info` output with OpenCode and Claude Code setup commands.

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
3. Downloads any missing GGUF weights from Hugging Face Hub (via `bartowski/*-GGUF` repos).
4. Generates a `llama-swap.yaml` configuration mapping each model to a llama-server backend (see [llama-swap Proxy Layer](#llama-swap-proxy-layer)).
5. Launches the `ghcr.io/thewulf7/localcode:cuda-latest` Docker container with:
   - GPU passthrough (`--gpus all`, with automatic CPU fallback if NVIDIA Container Toolkit is missing).
   - Volume mounts for models and config.
   - Port binding to the configured port (default `8080`).
   - All models preloaded on startup.

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

### `localcode info`

Display configuration instructions for connecting OpenCode and Claude Code to your running server.

```
localcode info
```

This command:
1. Loads your saved `localcode.json`.
2. Prints the OpenCode `config.json` snippet with correct model names and base URL.
3. Prints the Claude Code environment variables (`ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `CLAUDE_CODE_MAX_CONTEXT_TOKENS`).
4. Calculates `CLAUDE_CODE_MAX_CONTEXT_TOKENS` from your `ctx_size` (reserving ~15% for the model's response, minimum 4096 tokens).

Run this after `localcode init` or whenever you change your configuration to get copy-paste-ready commands for both clients.

---

## Configuration

### `localcode.json` Schema

```jsonc
{
  // Selected models with optional quantization override
  "models": [
    { "name": "Qwen/Qwen2.5-Coder-7B-Instruct", "quant": "Q8_0" },
    { "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "quant": "Q8_0" }
  ],

  // Whether to use Docker-based llama.cpp + llama-swap
  "run_in_docker": true,

  // Directory for downloaded GGUF weights (supports ~ expansion)
  "models_dir": "~/.opencode/models",

  // Port the LLM API binds to
  "port": 8080,

  // llama.cpp server arguments (auto-populated by init based on hardware)
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

### Project vs Global Scope

| Scope | Location | Created By |
|-------|----------|------------|
| **Local** (default) | `./localcode.json` in the working directory | `localcode init` |
| **Global** | `~/.config/localcode/localcode.json` | `localcode init --global` |

**Resolution order:** `localcode start` checks local first, then global. Local always wins.

### llama.cpp Server Arguments

The `llama_server_args` object is translated into CLI flags for the llama.cpp server process inside Docker:

| Key | CLI Flag | Description | Auto-configured |
|-----|----------|-------------|----------------|
| `ctx_size` | `--ctx-size` | Maximum context length in tokens | ✅ VRAM formula + quality cap |
| `n_gpu_layers` | `--n-gpu-layers` | Number of layers to offload to GPU (`999` = all) | ✅ Model size vs VRAM |
| `flash_attn` | `--flash-attn` | Flash attention mode (`on`/`off`) | ✅ Backend-aware (CUDA/Metal → on) |
| `cache_type_k` | `--cache-type-k` | KV cache quantization for keys (`q8_0`, `q4_0`, `f16`) | ✅ Headroom-aware |
| `cache_type_v` | `--cache-type-v` | KV cache quantization for values (`q8_0`, `q4_0`, `f16`) | ✅ Headroom-aware |
| `threads` | `--threads` | CPU threads for inference | ✅ Physical cores / 2 |
| `parallel` | `--parallel` | Concurrent request slots | ✅ VRAM budget / slot KV size |
| `mlock` | `--mlock` | Lock model in memory (prevent swap) | ✅ Linux only, when model fits |
| `slot-save-path` | `--slot-save-path` | Path to save/restore KV cache slots | ✅ Always `/models` |

**Any additional key-value pairs** in this object are passed through as `--key value` flags. Boolean `true` emits just the flag (`--mlock`); `false` is omitted. String/number values are appended as arguments.

> [!NOTE]
> The `flash_attn` field accepts string values `"on"` / `"off"` (current) or boolean `true` / `false` (legacy configs). Both formats are supported via automatic deserialization.

Example with extra args:

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

This produces:
```
--ctx-size 49152 --n-gpu-layers 999 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --threads 8 --parallel 2 --mlock --slot-save-path /models
```

Additionally, the `localcode start` command injects these **implicit flags** (not stored in `localcode.json`):

| Implicit Flag | Purpose |
|--------------|---------|
| `--jinja` | Use model's built-in chat template for tool calling |
| `--reasoning-format none` | Prevent misdetected reasoning format |
| `--rope-scaling yarn` | Enable YaRN context extension |
| `--override-kv {arch}.context_length=int:{ctx}` | Raise GGUF context cap (when ctx > 32768) |

### OpenCode Integration

During `localcode init`, an `opencode/config.json` file is automatically generated (or updated) with the correct provider URL and model names:

- **Local scope**: `./.opencode/config.json`
- **Global scope**: `~/.opencode/config.json`

The config sets up a `localcode` provider with model definitions, and places the active model selection at the root level:
- `model` (root) — the primary reasoning model (e.g., `Qwen/Qwen2.5-Coder-7B-Instruct`)
- `small_model` (root) — the fast autocomplete model (e.g., `Qwen/Qwen2.5-Coder-1.5B-Instruct`), if a combo was selected during init

Both models point at `http://localhost:<port>/v1` and the llama-swap proxy routes requests based on the model name in the payload.

### Claude Code Integration

LocalCode's llama-swap proxy implements the Anthropic Messages API (`/v1/messages`), so Claude Code works natively — it connects to the same port as OpenCode.

Run `localcode info` to get copy-paste-ready commands, or set the environment variables manually:

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

| Variable | Purpose | Notes |
|----------|---------|-------|
| `ANTHROPIC_BASE_URL` | Points Claude Code at your local server | **Do not** include `/v1` — Claude Code appends its own path |
| `ANTHROPIC_API_KEY` | Required by the client but unused locally | Any non-empty string works (`sk-localcode`) |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Limits how many tokens Claude Code packs per request | Must be ≤ `ctx_size - response_headroom`. See [Context Token Alignment](#context-token-alignment-claude-code) |

> [!TIP]
> Run `localcode info` after `init` or any config change — it calculates and prints the exact `CLAUDE_CODE_MAX_CONTEXT_TOKENS` value based on your current `ctx_size`.

#### How It Works Under the Hood

Claude Code sends Anthropic-format requests (model IDs like `claude-sonnet-4-6`). The llama-swap proxy maps these aliases to your local model and translates the API format. See [llama-swap Proxy Layer](#llama-swap-proxy-layer) for full details.

---

## Hardware Profiling

LocalCode uses [`llmfit-core`](https://crates.io/crates/llmfit-core) (v0.4.8+) during `init` to detect a comprehensive hardware profile:

| Field | Source | Used For |
|-------|--------|----------|
| **VRAM** (GPU memory in GB) | `SystemSpecs::detect()` | Context size formula, GPU layer offload, KV cache quant, parallel slots |
| **RAM** (system memory in GB) | `SystemSpecs::detect()` | CPU-only context sizing, fallback inference |
| **CPU Cores** (logical) | `SystemSpecs::detect()` | `--threads` calculation |
| **GPU Name** | `SystemSpecs::detect()` | Informational display |
| **GPU Backend** | `SystemSpecs::detect()` | Flash attention support (`Cuda`, `Metal`, `Vulkan`, `Rocm`, `Sycl`) |
| **GPU Count** | `SystemSpecs::detect()` | Multi-GPU awareness |
| **Unified Memory** | `SystemSpecs::detect()` | Apple Silicon → always full GPU offload |

Based on these values, `llmfit-core` returns:

1. **Recommended Models** — Scored and ranked models that fit your hardware, with optimal quantization levels.
2. **Recommended Combos** — Pairs of a standard reasoning model + a small autocomplete model that fit simultaneously in VRAM.

The interactive picker prioritizes **coding-oriented** models (filtering by category). In `--yes` mode, the best coding combo is selected automatically.

### Auto-Configured llama.cpp Args

Every parameter in `llama_server_args` is calculated from the hardware profile — no manual tuning needed. Here's what `from_hardware()` computes:

#### Context Size (`ctx_size`)

Uses a dual-constraint formula (see [Context Token Alignment](#context-token-alignment-claude-code) for full details):

$$\text{ctx\_size} = \min\!\Big(\frac{\text{VRAM} \times 0.90 - \text{params\_b} \times \text{bpp} - 0.5}{0.000008 \times \text{params\_b} \times \text{kv\_mult}},\; \text{native\_ctx} \times \text{YaRN\_factor}\Big)$$

Rounded down to nearest 1024, clamped to [2048, 131072].

**CPU-only fallback:** ≥32 GB RAM → 8192, ≥16 GB → 4096, else 2048.

#### GPU Layer Offload (`n_gpu_layers`)

| Condition | Value | Explanation |
|-----------|-------|-------------|
| No GPU (VRAM < 1 GB) | `0` | Pure CPU inference |
| Unified memory (Apple Silicon) | `999` | Shared memory pool — all layers always accessible |
| Model fits in VRAM (model_mem + 1 GB headroom) | `999` | Full GPU offload |
| Model exceeds VRAM | `floor(total_layers × VRAM / model_mem)` | Partial offload — layers proportional to available VRAM |

`total_layers ≈ params_b × 4` (standard transformer architecture heuristic).

#### Flash Attention (`flash_attn`)

| GPU Backend | Value | Notes |
|-------------|-------|-------|
| CUDA | `on` | Well-supported in llama.cpp |
| Metal | `on` | Well-supported on macOS |
| Vulkan | `off` | Support varies; disabled for safety |
| ROCm | `off` | Support varies; disabled for safety |
| SYCL | `off` | Support varies; disabled for safety |
| No GPU | `off` | Not applicable |

#### KV Cache Quantization (`cache_type_k`, `cache_type_v`)

| Condition | Type | Rationale |
|-----------|------|-----------|
| GPU + model quant contains `8` + headroom > 4 GB | `q8_0` | High-quality KV matched to weight quant |
| GPU + tight VRAM | `q4_0` | Compressed KV to save memory |
| CPU only | `f16` | Best quality when RAM is plentiful |

Headroom = `VRAM - model_mem - 0.5 GB`.

#### Threads (`--threads`)

```
physical_cores = cpu_cores / 2   (assumes 2 HW threads per core)

GPU mode:   min(physical_cores, 8), minimum 2
CPU mode:   physical_cores - 1, minimum 2
```

GPU inference uses the CPU mainly for tokenization and HTTP handling, so more than 8 threads provides no benefit.

#### Parallel Slots (`--parallel`)

Each parallel slot reserves its own KV cache buffer:

```
slot_kv_gb    = 0.000008 × params_b × kv_mult × ctx_size
free_for_slots = VRAM × 0.85 - model_mem - 0.5
max_slots      = floor(free_for_slots / slot_kv_gb)
parallel       = clamp(max_slots, 1, 4)
```

Only emitted when `parallel > 1` and VRAM ≥ model_mem + 2 GB. Allows concurrent requests without reloading.

#### Memory Lock (`--mlock`)

Emitted (Linux only) when the model fits entirely in VRAM (`VRAM ≥ model_mem + 1 GB`). Prevents the OS from swapping the model out of memory. Not used on macOS (unified memory makes it unnecessary).

#### Slot Save Path (`--slot-save-path`)

Always set to `/models` to enable slot caching inside the Docker container.

### Native Context Lengths

The quality cap depends on each model family's training context length:

| Model Family | Native Context | Notes |
|--------------|---------------|-------|
| Qwen 2.5 (7B/14B) | 32,768 | |
| Qwen 2.5 (32B/72B) | 131,072 | |
| Qwen 3 (all sizes) | 40,960 | |
| Llama 3.1 / 3.2 / 3.3 | 131,072 | |
| Llama 3 (original) | 8,192 | |
| DeepSeek (V2/V3/R1) | 131,072 | |
| Gemma 2 | 8,192 | |
| Phi-3 Mini | 4,096 | |
| Phi-3 (other) | 131,072 | |
| Mistral / Codestral | 32,768 | |
| StarCoder 2 | 16,384 | |
| Unknown models | 32,768 | Conservative fallback |

**YaRN extension factors** (safe multiplier for rope scaling beyond training length):

| Model Size | Factor | Quality Cap Example (32k native) |
|------------|--------|----------------------------------|
| ≤ 7B | 1.5× | 49,152 |
| 8B – 20B | 2.0× | 65,536 |
| > 20B | 2.5× | 81,920 |

---

## llama-swap Proxy Layer

LocalCode uses [llama-swap](https://github.com/mostlygeek/llama-swap) as a reverse proxy in front of llama.cpp. The proxy handles model routing, hot-swapping, and API translation — all on a single port.

### Docker Image

```
ghcr.io/thewulf7/localcode:cuda-latest
```

This image bundles llama-swap + llama.cpp (llama-server build 8262, CUDA 12.8). On systems without NVIDIA Container Toolkit, LocalCode automatically falls back to `ghcr.io/mostlygeek/llama-swap:cpu`.

### Generated Configuration

`localcode start` generates a `llama-swap.yaml` in your models directory. The YAML maps each model to a llama-server command line:

```yaml
includeAliasesInList: true
sendLoadingState: true

models:
  "Qwen/Qwen2.5-Coder-7B-Instruct":
    cmd: >-
      llama-server --port ${PORT}
        --model /models/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf
        --host 0.0.0.0 --jinja --reasoning-format none
        --rope-scaling yarn
        --override-kv qwen2.context_length=int:49152
        --ctx-size 49152 --n-gpu-layers 999
        --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
        --threads 8 --parallel 2 --mlock
        --slot-save-path /models
    filters:
      strip_params: "temperature, top_k, top_p, repeat_penalty"
      setParams:
        tool_choice:
          type: "any"
    aliases:
      - "claude-sonnet-4-6"
      - "claude-sonnet-4-5"
      - "claude-3-5-sonnet-latest"
      # ... all Claude model aliases

hooks:
  on_startup:
    preload:
      - "Qwen/Qwen2.5-Coder-7B-Instruct"
```

### Key Flags Explained

| Flag | Purpose |
|------|---------|
| `--jinja` | Use the model's built-in Jinja chat template for tool-call formatting |
| `--reasoning-format none` | Prevent llama-server from auto-detecting a reasoning format (e.g., "deepseek" for Qwen models) which disrupts grammar-constrained tool generation |
| `--rope-scaling yarn` | Enable YaRN (Yet another RoPE extensioN) to extend context beyond the model's native training length |
| `--override-kv {arch}.context_length=int:{ctx}` | Override the GGUF metadata context cap so llama-server doesn't reject requests exceeding `n_ctx_train` |

### Proxy Filters

**`strip_params`** — Removes Claude Code's sampling parameters (`temperature`, `top_k`, `top_p`, `repeat_penalty`) before they reach the local model. This prevents the cloud-tuned defaults from degrading local inference quality.

**`setParams: tool_choice: { type: "any" }`** — Forces llama-server's grammar-constrained tool-call generation to be active from the first token. Without this, the grammar is "lazy" — it only triggers when the model starts with the correct tool-call prefix (e.g. `<tool_call>\n`). Small models often start with markdown or XML instead, bypassing the grammar entirely and producing raw text instead of structured `tool_use` blocks.

### Model Aliases

When a **combo** is configured (primary + autocomplete model), aliases are split for intelligent routing:

- **Sonnet / Opus aliases → primary model**: `claude-3-5-sonnet-*`, `claude-3-opus-*`, `claude-3-sonnet-*`, `claude-sonnet-4-*`, `claude-opus-4-*`
- **Haiku aliases → autocomplete/small model**: `claude-3-5-haiku-*`, `claude-3-haiku-*`, `claude-haiku-4-*`

This enables **subagent routing** — Claude Code spawns lightweight subagents (which use haiku model IDs) and those requests are served by the fast, small model. The primary model handles the main agentic loop (sonnet/opus IDs).

When only a single model is configured (no combo), all aliases — including haiku — point to the primary model.

### Dual-Model VRAM Budget

When two models are loaded simultaneously, `from_hardware()` subtracts the autocomplete model's memory footprint (weights + ~0.3 GB overhead for its KV cache/compute buffers) from the VRAM budget before calculating the primary model's context size, GPU layer offload, and parallel slot count. This ensures the primary model's KV cache doesn't compete with the secondary model for VRAM.

### Architecture Key Inference

The `--override-kv` flag requires the GGUF architecture prefix. LocalCode infers it from the model name:

| Model Family | Architecture Key | Used In Override |
|--------------|-----------------|------------------|
| Qwen / Qwen2.5 / Qwen3 | `qwen2` | `qwen2.context_length` |
| Llama / CodeLlama | `llama` | `llama.context_length` |
| Mistral / Codestral | `mistral` | `mistral.context_length` |
| Gemma / CodeGemma | `gemma2` | `gemma2.context_length` |
| Phi-3 / Phi-3.5 | `phi3` | `phi3.context_length` |
| StarCoder 2 | `starcoder2` | `starcoder2.context_length` |
| DeepSeek | `deepseek2` | `deepseek2.context_length` |
| Command-R | `command-r` | `command-r.context_length` |

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

## Context Token Alignment (Claude Code)

When using Claude Code with a local model, you **must** set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to match your model's actual context window. Claude Code uses this value to decide how much conversation history, system prompt, and tool definitions to include in each request.

If this value is too high (or not set), Claude Code will pack requests that exceed the model's `ctx_size`, causing:
```
{"code":400,"message":"request (34588 tokens) exceeds the available context size (32768 tokens)"}
```

### How `ctx_size` Is Calculated from VRAM

LocalCode automatically calculates the optimal `ctx_size` during `localcode init` using two constraints:

**1. VRAM Budget** — what physically fits in GPU memory (llmfit-core formula):

$$\text{vram\_ctx} = \frac{\text{effective\_vram} \times 0.90 - \text{params\_b} \times \text{bpp} - 0.5}{0.000008 \times \text{params\_b} \times \text{kv\_mult}}$$

Where `effective_vram = VRAM − secondary_model_mem`. When a combo is configured, the autocomplete model's weight memory plus ~0.3 GB overhead is subtracted from the total VRAM before computing the primary model's context budget. For single-model setups, `effective_vram = VRAM`.

**2. Quality Ceiling** — what the model can handle without attention degradation:

$$\text{quality\_cap} = \text{native\_ctx} \times \text{YaRN\_factor}$$

The final `ctx_size = min(vram_ctx, quality_cap)`, rounded down to nearest 1024 and clamped to [2048, 131072].

| Variable | Meaning | Examples |
|----------|---------|----------|
| `params_b` | Model parameters in billions | 7.0 for Qwen 7B, 14.0 for Qwen 14B |
| `bpp` | Bytes per parameter for weight quant | Q8_0 = 1.05, Q4_K_M = 0.58 |
| `kv_mult` | KV cache multiplier vs f16 | q8_0 = 0.5, q4_0 = 0.25, f16 = 1.0 |
| `native_ctx` | Model's training context length | 32768 (Qwen 7B), 131072 (Llama 3.1) |
| `YaRN_factor` | Safe rope extension by model size | ≤7B = 1.5×, 8-20B = 2.0×, >20B = 2.5× |

**Example:** Qwen 7B Q8_0 on 16 GB VRAM with q8_0 KV cache:

```
VRAM budget:
  model_weights = 7.0 × 1.05 = 7.35 GB
  usable VRAM   = 16 × 0.90  = 14.4 GB
  free for KV   = 14.4 − 7.35 − 0.5 = 6.55 GB
  KV per token  = 0.000008 × 7.0 × 0.5 = 0.000028 GB
  vram_ctx      = 6.55 / 0.000028 ≈ 233,928

Quality ceiling:
  native_ctx    = 32,768 (Qwen 2.5 7B)
  YaRN_factor   = 1.5 (≤7B model)
  quality_cap   = 32,768 × 1.5 = 49,152

Final: min(233928, 49152) = 49,152  ← quality is the bottleneck, not VRAM
```

> [!NOTE]
> **For Claude Code usage**, VRAM is rarely the bottleneck — the model's training context × YaRN multiplier is. A 7B model on 16GB has plenty of VRAM headroom but can only reliably handle ~49k tokens. To get larger context, use a model family with a larger native window (e.g., Llama 3.1 at 128k, DeepSeek at 128k, or Qwen 72B at 128k).

### How to Calculate `CLAUDE_CODE_MAX_CONTEXT_TOKENS`

```
CLAUDE_CODE_MAX_CONTEXT_TOKENS = ctx_size - response_headroom
```

**Response headroom** is the space reserved for the model's output. Use ~15% of `ctx_size`, with a minimum of 4096 tokens:

| `ctx_size` in localcode.json | `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Headroom |
|------------------------------|--------------------------------|----------|
| 8192                         | 4096                           | 4096     |
| 16384                        | 12288                          | 4096     |
| 32768                        | 28672                          | 4096     |
| 49152                        | 42132                          | 7020     |
| 65536                        | 56196                          | 9340     |

### Setting It

**macOS / Linux:**
```bash
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=42132
```

**Windows (PowerShell):**
```powershell
$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS=42132
```

> [!TIP]
> Run `localcode info` after init or any config change — it calculates and prints the correct `CLAUDE_CODE_MAX_CONTEXT_TOKENS` value based on your current `ctx_size`.

### Why It Matters

Claude Code's system prompt + tool definitions alone consume ~15-20k tokens. A typical coding session with file contents can easily reach 30-40k tokens. If `ctx_size` is 32768 (a common default for 7B models), you're already at the edge.

If you've set a large `ctx_size` (e.g., 49152 or 65536 with `--rope-scaling yarn`), make sure `CLAUDE_CODE_MAX_CONTEXT_TOKENS` reflects it — otherwise Claude Code defaults to a much smaller window, leaving your extended context unused.

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
