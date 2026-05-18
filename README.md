# P-EAGLE

### ⚡ Parallel Speculative Decoding Framework for LLM Inference

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Achieve 1.5-2.5x speedup** on LLM inference by predicting multiple future tokens in parallel.

</div>

---

## 🚀 Quick Start

### Install & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (data → features → training → evaluation)
./run_full_pipeline.sh
```

### Or Use Individual Scripts

| Stage | Command |
|-------|---------|
| **Feature Extraction** | `python -m p_eagle.scripts.extract_features ...` |
| **Single-GPU Training** | `./automation.sh single` |
| **Multi-Node Training** | `./automation.sh multi` |
| **Evaluation** | `python -m p_eagle.scripts.evaluate ...` |

---

## 🏗️ Architecture

P-EAGLE implements **EAGLE-3** architecture with hidden state injection for parallel speculative decoding.

```
┌─────────────────────────────────────────────────────────────┐
│                      TARGET MODEL                           │
│                   (e.g., Gemma-3-4B, 2560 dim)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DRAFTER MODEL                          │
│                   (e.g., Gemma-3-270M, 640 dim)            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  EAGLE-3: First layer accepts [embeddings ⊕ hidden] │    │
│  │  → Produces K parallel token predictions             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│         h_{t+1}    h_{t+2}    h_{t+3}    h_{t+K}           │
│            ▲         ▲         ▲           ▲                │
│            │         │         │           │                │
│        MTP Head 1  MTP Head 2 MTP Head 3 ...MTP Head K     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Tree Attention │
                    │  Verify K tokens│
                    │   in 1 pass     │
                    └─────────────────┘
```

**Key Innovations:**
- ✨ **EAGLE-3 Hidden Injection** - Concatenate target hidden states with embeddings
- 🎯 **Multi-Token Prediction** - K parallel heads predict future tokens
- 🌳 **Tree Attention** - Verify K tokens in single forward pass
- 🔥 **Multi-Node Training** - Scale across 2+ machines with DDP

---

## ⚙️ Configuration

Edit `automation.sh` to customize training:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPECULATION_DEPTH` | `4` | Number of MTP heads (K) |
| `MAX_SEQ_LEN` | `2048` | Max tokens per sequence |
| `BATCH_SIZE` | `16` | Training batch size |
| `NUM_EPOCHS` | `5` | Training epochs |
| `LEARNING_RATE` | `1e-5` | Learning rate |
| `TARGET_HIDDEN_DIM` | `2560` | Target model hidden dim |

### Multi-Node Setup

```bash
# Machine 1 (Master) - 192.168.201.2
./automation.sh multi

# Machine 2 (Worker) - 192.168.201.3
./automation.sh multi
```

---

## 💻 Hardware Requirements

| Drafter | VRAM | Training Time (7k samples) |
|---------|------|----------------------------|
| 270M | ~12 GB | ~2-3 hours/epoch |
| 1B | ~20 GB | ~4-6 hours/epoch |

---

## 📁 Project Structure

```
p_eagle/
├── p_eagle/                      # Main package
│   ├── models/                   # Model definitions
│   │   ├── peagle_drafter.py   # P-EAGLE Drafter + MTP heads
│   │   └── flash_attention.py   # Flash attention
│   ├── training/                 # Training
│   │   ├── trainer.py           # Training loop (DDP support)
│   │   └── feature_extractor.py # Feature extraction
│   ├── scripts/                  # CLI entry points
│   └── utils/                    # Utilities
├── data/                         # Data
│   └── features/                # Extracted features (.pt)
├── checkpoints/                  # Model checkpoints
│   └── automated_multi/         # Multi-node output
├── automation.sh                 # Training automation
├── run_full_pipeline.sh         # Full pipeline
└── requirements.txt
```

---

## 🔧 Troubleshooting

### OOM During Training?
```bash
# Reduce in automation.sh
BATCH_SIZE=8
MAX_SEQ_LEN=1024
```

### Training Crashed?
```bash
# Check logs
cat checkpoints/automated_multi/logs/*/training.log

# Increase NCCL timeout
export NCCL_TIMEOUT=7200
```

### Slow Training?
- Data loading is typically the bottleneck
- Ensure fast disk I/O for feature files
- Consider NVMe storage

---

## 📚 Citation

```bibtex
@article{eagle2024,
  title={EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty},
  author={Li, Xiaotian and Liu, Zheng and others},
  journal={arXiv preprint},
  year={2024}
}
```

---

<div align="center">

**Made with ⚡ for fast LLM inference**

</div>