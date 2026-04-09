# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nunchaku is a high-performance inference engine for 4-bit quantized neural networks, implementing **SVDQuant** (post-training quantization for diffusion models). It combines a Python frontend (diffusers-compatible) with optimized CUDA/C++ kernels for W4A4 (4-bit weight + 4-bit activation) inference.

## Build & Install

**Requirements**: Python >= 3.10, CUDA >= 12.2 (Linux) / >= 12.6 (Windows), gcc/g++ >= 11, PyTorch >= 2.7

```bash
# Development install (compiles CUDA kernels for detected GPU)
pip install -e ".[dev]"

# Clone with submodules (cutlass, json, mio, spdlog, Block-Sparse-Attention)
git clone --recurse-submodules <repo-url>
```

`NUNCHAKU_INSTALL_MODE=FAST` (default) compiles only for the detected GPU architecture. Set `NUNCHAKU_INSTALL_MODE=ALL` for all supported architectures (sm_75, sm_80, sm_86, sm_89, sm_120, sm_121).

## Linting

```bash
pre-commit install          # one-time setup
pre-commit run --all-files  # run all checks
```

Linting stack: **ruff**, **black**, **isort** (all line-length=120), **clang-format** (LLVM style, 120 cols) for C++/CUDA, plus nbstripout, yamlfmt, mdformat, doc8, rstcheck.

## Testing

```bash
export HF_TOKEN=$YOUR_HF_TOKEN  # required for model downloads

# Run specific test suites
pytest -v tests/flux/test_flux_examples.py
pytest -v tests/flux --ignore=tests/flux/test_flux_memory.py
pytest -v tests/sana
pytest -v tests/v1

# Run all tests
python .github/workflows/run_all_tests.py
```

Tests use LPIPS metric to compare generated images against references. Set `NUNCHAKU_TEST_CACHE_ROOT` to cache reference images across runs.

## Architecture

### Python Package (`nunchaku/`)

- **`models/`** — Core model implementations:
  - `transformers/` — FLUX, SANA, Qwen-Image, Z-Image transformer blocks
  - `text_encoders/` — Quantized T5 and CLIP encoders
  - `linear.py` — 4-bit quantized linear layer (Python wrapper over C++ kernels)
  - `unets/`, `attention.py`, `embeddings.py`, `normalization.py`
  - `ip_adapter/` — IP-Adapter support
- **`lora/`** — LoRA support for FLUX models
- **`caching/`** — KV-cache and activation caching (FB Cache, Double FB Cache)
- **`ops/`** — Custom operations (GEMM, quantization)
- **`pipeline/`** — Pipeline wrappers for diffusers integration
- **`csrc/pybind.cpp`** — PyBind11 bindings exposing `nunchaku._C`

### C++/CUDA Source (`src/`)

- `FluxModel.cpp/h`, `SanaModel.cpp/h` — Model-level C++ implementations
- `Linear.cpp/h` — 4-bit linear layer kernel dispatch
- `kernels/` — CUDA kernels:
  - `zgemm/gemm_w4a4.cu` — W4A4 GEMM (core compute kernel)
  - Activation, layernorm, block-sparse attention kernels
- `Module.h`, `Tensor.h` — Base abstractions

### Data Flow

User Python code → diffusers pipeline → Nunchaku model (`from_pretrained`) → Python model layers → C++ extension (`nunchaku._C`) → CUDA kernels (W4A4 GEMM with fused quantize/dequantize)

### Exported Models (`nunchaku/__init__.py`)

`NunchakuFluxTransformer2dModel`, `NunchakuFluxTransformer2DModelV2`, `NunchakuSanaTransformer2DModel`, `NunchakuQwenImageTransformer2DModel`, `NunchakuZImageTransformer2DModel`, `NunchakuT5EncoderModel`

## PR Guidelines

- Create feature branches (`feat/my-feature`), never commit directly to main
- All pre-commit checks must pass
- Include tests for new features; do not modify existing tests
- CI runs on self-hosted runners with RTX 5090 and RTX 4090
