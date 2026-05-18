# P-EAGLE Project Structure

This document describes the organization of the P-EAGLE codebase.

## Directory Overview

```
P_Eagle/
│
├── p_eagle/                          # Main Python package
│   ├── __init__.py                    # Package initialization
│   │
│   ├── models/                        # Neural network architectures
│   │   ├── __init__.py
│   │   ├── peagle_drafter.py         # Main drafter with EAGLE-3 architecture
│   │   ├── flash_attention.py        # Flash attention implementation
│   │   └── tree_attention.py         # Tree attention for parallel verification
│   │
│   ├── training/                      # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training loop (EagleTrainer)
│   │   └── feature_extractor.py      # Hidden state extraction from target
│   │
│   ├── inference/                     # Inference engine
│   │   ├── __init__.py
│   │   └── inference_engine.py       # Speculative decoding engine
│   │
│   ├── scripts/                      # CLI entry points
│   │   ├── __init__.py
│   │   ├── extract_features.py       # Wrapper for feature extraction
│   │   ├── train_drafter.py          # Wrapper for training
│   │   ├── run_inference.py          # Wrapper for inference
│   │   └── evaluate.py               # Evaluation script
│   │
│   └── utils/                         # Helper utilities
│       ├── __init__.py
│       ├── feature_utils.py          # Feature processing utilities
│       └── loss_utils.py            # Loss function implementations
│
├── scripts/                           # Standalone utilities
│   ├── generate_data.py              # Synthetic data generation
│   ├── preflight_check.py            # System requirements check
│   └── sync_to_worker.sh             # Sync script for multi-GPU
│
├── plot_scripts/                      # Visualization scripts
│   ├── __init__.py
│   ├── generate_plots.py             # Main plot generation
│   ├── plot_training.py             # Training curves
│   ├── plot_evaluation.py           # Evaluation results
│   ├── plot_comparison.py           # Model comparison
│   ├── utils.py                     # Plotting utilities
│   └── plots/                       # Output directory for plots
│
├── data/                              # Data storage (gitignored)
│   ├── raw/                          # Raw input data
│   ├── processed/                    # Processed datasets
│   ├── features/                    # Extracted hidden states
│   └── output/                      # Generated datasets
│
├── checkpoints/                      # Model checkpoints (gitignored)
├── models_cache/                     # HuggingFace model cache (gitignored)
├── logs/                             # Training logs (gitignored)
│
├── run_full_pipeline.sh              # Main automation script
├── setup.py                          # Package installation
├── requirements.txt                  # Python dependencies
├── README.md                         # Main documentation
├── ARCHITECTURE.md                   # Architecture deep dive
├── TRAINING.md                       # Training guide
│
└── venv/                             # Python virtual environment (gitignored)
```

## Core Modules

### 1. Models (`p_eagle/models/`)

| File | Class/Function | Purpose |
|------|----------------|---------|
| `peagle_drafter.py` | `PEagleDrafterModel` | Main drafter with LoRA and EAGLE-3 |
| `peagle_drafter.py` | `MTPHead` | Multi-token prediction head |
| `peagle_drafter.py` | `ProjectionLayer` | Hidden dimension projection |
| `flash_attention.py` | `FlashAttention` | Optimized attention implementation |
| `tree_attention.py` | `TreeAttentionMask` | Tree-structured attention mask |

**Usage:**
```python
from p_eagle.models import PEagleDrafterModel

model = PEagleDrafterModel(
    base_model_name="google/gemma-3-270m-it",
    target_hidden_dim=2560,
    speculation_depth=4,
    use_lora=True,
    lora_rank=64
)
```

### 2. Training (`p_eagle/training/`)

| File | Class/Function | Purpose |
|------|----------------|---------|
| `feature_extractor.py` | `FeatureExtractor` | Extracts hidden states from target |
| `feature_extractor.py` | `TriLayerConfig` | Layer selection configuration |
| `trainer.py` | `EagleTrainer` | Main training loop |
| `trainer.py` | `main()` | CLI entry point |

**Feature Extraction:**
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

**Training:**
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

### 3. Inference (`p_eagle/inference/`)

| File | Class | Purpose |
|------|-------|---------|
| `inference_engine.py` | `PEAGLEInference` | End-to-end speculative decoding |
| `inference_engine.py` | `SpeculationResult` | Result container with metrics |

**Inference:**
```python
from p_eagle.inference import PEAGLEInference

engine = PEAGLEInference(
    drafter_checkpoint="checkpoints/best_model",
    target_model="google/gemma-3-4b-it",
    speculation_depth=4
)
result = engine.generate("Hello, world!", max_tokens=100)
print(result.text)
```

### 4. Scripts (`p_eagle/scripts/`)

| File | Entry Point | Purpose |
|------|-------------|---------|
| `extract_features.py` | `p-eagle-extract` | CLI for feature extraction |
| `train_drafter.py` | `p-eagle-train` | CLI for training |
| `run_inference.py` | `p-eagle-infer` | CLI for inference |
| `evaluate.py` | - | Evaluation metrics calculation |

**CLI Usage:**
```bash
# Extract features
python -m p_eagle.scripts.extract_features --help

# Train drafter
python -m p_eagle.scripts.train_drafter --help

# Run inference
python -m p_eagle.scripts.run_inference --help

# Evaluate
python -m p_eagle.scripts.evaluate --help
```

### 5. Utilities (`p_eagle/utils/`)

| File | Functions | Purpose |
|------|-----------|---------|
| `feature_utils.py` | `fuse_features`, `normalize_hidden` | Feature processing |
| `loss_utils.py` | `kl_divergence`, `combined_loss` | Loss computations |

## Standalone Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `generate_data.py` | Generate synthetic training data |
| `preflight_check.py` | Validate system requirements |
| `sync_to_worker.sh` | Sync project to worker nodes |

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT DATA FLOW                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │ ──► │  Processed   │ ──► │  Features    │
│  (JSONL)     │     │  (JSONL)     │     │  (.pt shards)│
└──────────────┘     └──────────────┘     └──────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
generate_data.py      generate_data.py    feature_extractor
      │                    │                    │
      └────────────────────┴────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Training    │
                    │  (.pt files)  │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Checkpoints │
                    │  (best_model)│
                    └──────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Inference│    │ Evaluate │    │  Plots   │
    └──────────┘    └──────────┘    └──────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_MODEL` | `google/gemma-3-4b-it` | Target model for feature extraction |
| `DRAFTER_MODEL` | `google/gemma-3-270m-it` | Drafter model for training |
| `SPECULATION_DEPTH` | `1` | Number of MTP heads |
| `LORA_RANK` | `16` | LoRA adaptation rank |
| `BATCH_SIZE` | `4` | Training batch size |
| `EPOCHS` | `1` | Number of training epochs |
| `LEARNING_RATE` | `2e-5` | Learning rate |

## Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `setup.py` | Package installation and entry points |
| `run_full_pipeline.sh` | Full pipeline automation |

## Quick Commands

```bash
# Full pipeline (single GPU)
./run_full_pipeline.sh

# Multi-GPU training
./run_full_pipeline.sh multi <master_ip>

# DeepSpeed training
./run_full_pipeline.sh deepspeed <master_ip>

# Individual steps
python -m p_eagle.scripts.extract_features --model_path google/gemma-3-4b-it --input_data data/raw/train.jsonl --output_dir data/features
python -m p_eagle.scripts.train_drafter --drafter_model google/gemma-3-270m-it --target_hidden_dim 2560 --feature_dir data/features --output_dir checkpoints
python -m p_eagle.scripts.evaluate --drafter_checkpoint checkpoints/best_model --target_model google/gemma-3-4b-it
```

## Dependencies

```
torch>=2.0.0           # PyTorch deep learning framework
transformers>=4.36.0   # HuggingFace transformers
peft>=0.7.0            # Parameter-efficient fine-tuning
accelerate>=0.25.0     # Training acceleration
bitsandbytes>=0.41.0   # Quantization support
```