# P-EAGLE

**Parallel Speculative Decoding Framework for LLM Inference**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Flash Attention](https://img.shields.io/badge/Flash%20Attention-Enabled-blueviolet.svg)](https://arxiv.org/abs/2205.14135)
[![Multi-GPU](https://img.shields.io/badge/Multi--GPU-DDP-orange.svg)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
[![LoRA](https://img.shields.io/badge/LoRA-Supported-green.svg)](https://arxiv.org/abs/2106.09685)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Achieve 1.5-2.5x inference speedup** through parallel speculative decoding with multi-token prediction.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Speculative Decoding](#speculative-decoding)
- [Configuration](#configuration)
- [Hardware Requirements](#hardware-requirements)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

P-EAGLE (Parallel EAGLE) is a high-performance speculative decoding framework that accelerates large language model inference by 1.5-3x without quality degradation. It implements the EAGLE-3 architecture with hidden state injection and multi-token prediction heads.

**Key Idea:** Instead of generating tokens sequentially, P-EAGLE uses a lightweight "drafter" model to predict multiple future tokens in parallel. A "tree attention" mechanism then verifies all predictions in a single forward pass through the target model.

```
Traditional (Autoregressive):     P-EAGLE (Speculative):
t1 → t2 → t3 → t4 → t5 → ...     t1 → [t2,t3,t4,t5] → verify all at once
    (5 steps)                          (2 steps = 2.5x speedup)
```

---

## Features

| Feature | Description |
|---------|-------------|
| **EAGLE-3 Architecture** | Novel hidden state injection combining target embeddings with hidden states |
| **Multi-Token Prediction** | K parallel prediction heads for speculating multiple future tokens |
| **Tree Attention** | O(1) verification complexity for K speculative tokens |
| **Flash Attention** | Memory-efficient attention implementation for large contexts |
| **Multi-GPU Training** | Distributed Data Parallel (DDP) for scaling across multiple GPUs |
| **LoRA Adaptation** | Parameter-efficient fine-tuning for quick drafter customization |
| **Flexible Layer Selection** | Extract and fuse features from any combination of model layers |

**Supported Models:**
- Gemma-3 (270M, 1B, 4B)
- Llama-3 (1B, 8B)
- Custom transformers with hidden states

---

## Installation

### Prerequisites

```bash
# Python 3.10+
python --version  # >= 3.10

# PyTorch 2.0+ with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 11.8+
nvcc --version  # >= 11.8
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-repo/p-eagle.git
cd p-eagle

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Requirements

```
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
flash-attn>=2.0.0
```

---

## Quick Start

### Run Full Pipeline

```bash
# Complete pipeline: data generation → feature extraction → training → evaluation
./run_full_pipeline.sh
```

### Step-by-Step

```bash
# 1. Generate training data
python scripts/generate_data.py \
    --local \
    --num-samples 5000 \
    --input-dir data/raw \
    --output data/processed

# 2. Extract hidden states from target model
python -m p_eagle.scripts.extract_features \
    --model_path google/gemma-3-4b-it \
    --input_data data/processed/*.jsonl \
    --output_dir data/features \
    --layers -1,-2,-3 \
    --fusion mean

# 3. Train the drafter model
python -m p_eagle.scripts.train_drafter \
    --drafter_model google/gemma-3-270m-it \
    --target_hidden_dim 2560 \
    --feature_dir data/features \
    --output_dir checkpoints \
    --use_lora \
    --lora_rank 64 \
    --epochs 5

# 4. Evaluate performance
python -m p_eagle.scripts.evaluate \
    --drafter_checkpoint checkpoints/best_model \
    --target_model google/gemma-3-4b-it
```

### Interactive Animation

Visualize the EAGLE-3 architecture, data pipeline, and speculative decoding process:

```bash
# Full animation
python scripts/animate_workflow.py --all

# Individual sections
python scripts/animate_workflow.py --arch         # Architecture diagram
python scripts/animate_workflow.py --workflow     # Data pipeline
python scripts/animate_workflow.py --speculative  # Speculative decoding
```

---

## Architecture

### EAGLE-3 Innovation

P-EAGLE implements EAGLE-3, a novel speculative decoding architecture:

1. **Hidden State Injection**: The drafter receives both token embeddings AND hidden states from the target model, enabling it to predict more accurately.

2. **Multi-Token Prediction**: K parallel heads predict the next K tokens simultaneously, rather than just the next token.

3. **Tree Attention**: All K speculative tokens are verified in a single forward pass, achieving O(1) verification complexity per token.

### Key Components

| Component | Description | Size |
|-----------|-------------|------|
| Target Model | Main LLM providing hidden states | 4B-8B |
| Drafter Model | Lightweight predictor with EAGLE-3 | 270M-1B |
| Feature Extractor | Extracts hidden states from target | - |
| MTP Heads | K parallel prediction heads | ~1M params |
| Tree Attention | Parallel verification mechanism | - |

### Dimensionality

```
Target Model:
  - Hidden dim: 2560 (4B) or 4096 (8B)
  - Layers: 18-32

Drafter Model:
  - Hidden dim: 640
  - Layers: 6-12
  - First layer projects [embeddings ⊕ hidden] → 640 dims

EAGLE-3 Injection:
  - Input: [token_embedding (4096) ⊕ target_hidden (2560)] = 6656 dims
  - Output: 640 dims (drafter hidden dimension)
```

---

## Data Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐
│  RAW DATA   │───▶│  PROCESSED  │───▶│  FEATURES   │───▶│     TRAINING     │
│   JSONL     │    │    JSONL    │    │  .pt shards  │    │     .pt files     │
└─────────────┘    └─────────────┘    └─────────────┘    └────────┬─────────┘
      │                  │                   │                    │
      ▼                  ▼                   ▼                    ▼
generate_data.py   generate_data.py   feature_extractor   EagleTrainer
```

### Pipeline Stages

| Stage | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| 1 | `generate_data.py` | Raw JSONL | Processed JSONL | Tokenize, filter, deduplicate |
| 2 | `generate_data.py` | - | - | Additional formatting |
| 3 | `extract_features` | Processed JSONL | `.pt` shards | Extract hidden states |
| 4 | `train_drafter` | `.pt` shards | `checkpoints/` | Train EAGLE-3 drafter |
| 5 | Evaluate | Checkpoint | Metrics | Benchmark performance |

---

## Speculative Decoding

P-EAGLE accelerates inference through speculative decoding:

### Process

```
┌─────────────────────────┐
│  1. DRAFTER GENERATES   │
│  K tokens in parallel   │
│  [the][cat][sat][on]    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  2. TREE ATTENTION      │
│  Verifies all K tokens  │
│  in ONE forward pass    │
│                         │
│  Draft: [the][cat][sat][on]
│  Target:[the][cat][sat][mat]
│  Match:   ✓   ✓   ✓   ✗
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  3. ACCEPT/REJECT       │
│  Accept: [the][cat][sat]
│  Resample: [on] → [mat] │
│                         │
│  Result: 3 tokens in    │
│  time of 1 step!        │
└─────────────────────────┘
```

### Acceptance Criteria

| Token | Match? | Action |
|-------|--------|--------|
| `the` | ✓ | ACCEPT |
| `cat` | ✓ | ACCEPT |
| `sat` | ✓ | ACCEPT |
| `on` | ✗ | REJECT + RESAMPLE |

After first rejection, all subsequent tokens are also rejected.

### Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Acceptance Rate | >70% | % of drafted tokens accepted |
| Speedup | 1.5-3x | Tokens/second vs autoregressive |
| Memory Overhead | +270MB | Drafter model + KV cache |
| Output Quality | Identical | Probabilistically same as target |

---

## Configuration

### Training Parameters

```bash
# Via command line
python -m p_eagle.scripts.train_drafter \
    --drafter_model google/gemma-3-270m-it \
    --target_hidden_dim 2560 \
    --speculation_depth 4 \
    --max_seq_len 2048 \
    --batch_size 16 \
    --num_epochs 5 \
    --learning_rate 1e-5 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --gradient_checkpointing
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--drafter_model` | required | Base model for drafter |
| `--target_hidden_dim` | 2560 | Target model hidden dimension |
| `--speculation_depth` | 4 | Number of MTP heads (K) |
| `--max_seq_len` | 2048 | Maximum sequence length |
| `--batch_size` | 16 | Training batch size |
| `--num_epochs` | 5 | Number of training epochs |
| `--learning_rate` | 1e-5 | Optimizer learning rate |
| `--use_lora` | false | Use LoRA adaptation |
| `--lora_rank` | 64 | LoRA rank |
| `--lora_alpha` | 128 | LoRA scaling factor |

### Feature Extraction Parameters

```bash
python -m p_eagle.scripts.extract_features \
    --model_path google/gemma-3-4b-it \
    --input_data data/processed/*.jsonl \
    --output_dir data/features \
    --layers -1,-2,-3 \
    --fusion mean \
    --batch_size 8 \
    --max_length 2048
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | required | Target model path |
| `--layers` | `-1,-2,-3` | Layers to extract |
| `--fusion` | `mean` | Layer fusion method (mean/concat) |
| `--batch_size` | 8 | Extraction batch size |
| `--max_length` | 2048 | Maximum sequence length |

### Multi-GPU Training

```bash
# Machine 1 (Master)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=2
python -m p_eagle.scripts.train_drafter \
    --multi_gpu \
    --local_rank 0 \
    ...

# Machine 2 (Worker)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=2
python -m p_eagle.scripts.train_drafter \
    --multi_gpu \
    --local_rank 1 \
    ...
```

Or use the automation script:

```bash
./automation.sh multi
```

---

## Hardware Requirements

### GPU Memory

| Model Size | Training VRAM | Inference Overhead |
|------------|---------------|-------------------|
| 270M drafter | ~12 GB | +270 MB |
| 1B drafter | ~20 GB | +1 GB |

### Recommended Setup

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24GB) | A100 (40-80GB) |
| CPU | 8 cores | 16+ cores |
| RAM | 32 GB | 64 GB |
| Storage | SSD | NVMe SSD |

### Training Time Estimates (7,000 samples)

| Drafter | VRAM | Single GPU | Multi-GPU (2x) |
|---------|------|------------|----------------|
| 270M | 12 GB | ~2-3 hours/epoch | ~1-1.5 hours/epoch |
| 1B | 20 GB | ~4-6 hours/epoch | ~2-3 hours/epoch |

---

## API Reference

### Python API

#### FeatureExtractor

```python
from p_eagle.training import FeatureExtractor

extractor = FeatureExtractor(
    model_name="google/gemma-3-4b-it",
    output_dir="data/features",
    layers=[-1, -2, -3],
    fusion="mean"
)
extractor.extract_features("data/processed/*.jsonl")
```

#### EagleTrainer

```python
from p_eagle.training import EagleTrainer

trainer = EagleTrainer(
    drafter_model_name="google/gemma-3-270m-it",
    target_hidden_dim=2560,
    feature_dir="data/features",
    output_dir="checkpoints",
    use_lora=True,
    lora_rank=64
)
trainer.train()
```

#### PEAGLEInference

```python
from p_eagle.inference import PEAGLEInference

engine = PEAGLEInference(
    drafter_checkpoint="checkpoints/best_model",
    target_model="google/gemma-3-4b-it",
    speculation_depth=4
)
result = engine.generate("Explain quantum computing:", max_tokens=100)
print(result.text)
```

### CLI Commands

```bash
# Extract features
p-eagle-extract --help
p-eagle-extract --model_path google/gemma-3-4b-it --input_data data/*.jsonl

# Train drafter
p-eagle-train --help
p-eagle-train --drafter_model google/gemma-3-270m-it --feature_dir data/features

# Run inference
p-eagle-infer --help
p-eagle-infer --drafter_checkpoint checkpoints/best_model --prompt "Hello"
```

---

## Examples

### Basic Training

```python
from p_eagle.training import EagleTrainer

trainer = EagleTrainer(
    drafter_model_name="google/gemma-3-270m-it",
    target_hidden_dim=2560,
    feature_dir="data/features",
    output_dir="checkpoints",
    use_lora=True,
    lora_rank=64,
    batch_size=16,
    num_epochs=5
)

trainer.train()
print(f"Model saved to: {trainer.output_dir}/best_model")
```

### Speculative Inference

```python
from p_eagle.inference import PEAGLEInference

engine = PEAGLEInference(
    drafter_checkpoint="checkpoints/best_model",
    target_model="google/gemma-3-4b-it",
    speculation_depth=4
)

# Generate with speculative decoding
result = engine.generate(
    "Write a Python function to sort a list:",
    max_tokens=200,
    temperature=0.7
)

print(f"Generated: {result.text}")
print(f"Acceptance rate: {result.acceptance_rate:.1%}")
print(f"Speedup: {result.speedup:.2f}x")
```

### Batch Inference

```python
from p_eagle.inference import PEAGLEInference

engine = PEAGLEInference(
    drafter_checkpoint="checkpoints/best_model",
    target_model="google/gemma-3-4b-it",
)

prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is transformers architecture?",
]

results = engine.batch_generate(prompts, max_tokens=100)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result.text}\n")
```

---

## Troubleshooting

### Out of Memory During Training

```bash
# Reduce batch size
--batch_size 8

# Reduce sequence length
--max_seq_len 1024

# Enable gradient checkpointing
--gradient_checkpointing

# Use mixed precision
--fp16
```

### Slow Training

- **Data loading bottleneck**: Use NVMe storage for feature files
- **GPU utilization**: Check CUDA_VISIBLE_DEVICES is set correctly
- **Pre-extract features**: Run feature extraction once, reuse cached features

### Low Acceptance Rate

- Train for more epochs (try 10 instead of 5)
- Increase LoRA rank (try 128 instead of 64)
- Use more layers for feature extraction (add -4, -5)
- Check data quality (ensure clean, diverse training data)

### Multi-GPU Issues

```bash
# Increase NCCL timeout
export NCCL_TIMEOUT=7200

# Debug with single GPU first
./automation.sh single

# Check GPU connectivity
python -c "import torch; print(torch.cuda.device_count())"
```

### Common Errors

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size or use gradient checkpointing |
| `RuntimeError: NCCL timeout` | Increase NCCL_TIMEOUT or check network |
| `ModuleNotFoundError: peft` | `pip install peft` |
| `ValueError: model not found` | Check HuggingFace model name/path |

---

## Project Structure

```
P_Eagle/
├── p_eagle/                          # Main package
│   ├── models/                       # Model definitions
│   │   ├── peagle_drafter.py         # P-EAGLE Drafter + MTP heads
│   │   ├── flash_attention.py        # Flash attention
│   │   └── tree_attention.py         # Tree attention mask
│   ├── training/                     # Training pipeline
│   │   ├── trainer.py                # EagleTrainer
│   │   └── feature_extractor.py      # Hidden state extraction
│   ├── inference/                    # Inference engine
│   │   ├── inference_engine.py       # Speculative decoding
│   │   └── tree_mask.py              # Tree attention utilities
│   ├── scripts/                      # CLI entry points
│   │   ├── extract_features.py
│   │   ├── train_drafter.py
│   │   └── evaluate.py
│   └── utils/                        # Utilities
│       ├── feature_utils.py
│       └── loss_utils.py
├── scripts/                          # Standalone utilities
│   ├── generate_data.py              # Data generation
│   ├── preflight_check.py            # System check
│   ├── sync_to_worker.sh             # Multi-GPU sync
│   └── animate_workflow.py           # Architecture animation
├── data/                             # Data storage (gitignored)
├── checkpoints/                      # Model checkpoints (gitignored)
├── tests/                            # Unit tests
├── run_full_pipeline.sh              # Main pipeline
├── automation.sh                      # Training automation
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ⚡ for fast LLM inference**

</div>