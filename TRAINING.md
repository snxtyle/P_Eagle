# P-EAGLE Training Guide

## 🚀 Quick Start

### Single Node (1 GPU)
```bash
./run_full_pipeline.sh
```

### Multi-Node (2 GPUs across 2 machines)
```bash
# On MASTER (192.168.1.100):
./run_full_pipeline.sh multi 192.168.1.100

# On WORKER (192.168.1.101):
./run_full_pipeline.sh multi 192.168.1.100
```

### DeepSpeed ZeRO-3
```bash
./run_full_pipeline.sh deepspeed <master_ip>
```

---

## ⚙️ Configuration

### Environment Variables (override defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `DRAFTER_MODEL` | `google/gemma-3-270m-it` | Base drafter model |
| `TARGET_HIDDEN_DIM` | `2560` | Target model hidden dim |
| `SPECULATION_DEPTH` | `1` | Number of MTP heads (K) |
| `MAX_SEQ_LEN` | `2048` | Max sequence length |
| `BATCH_SIZE` | `4` | Training batch size |
| `EPOCHS` | `1` | Number of epochs |
| `LEARNING_RATE` | `2e-5` | Learning rate |
| `LORA_RANK` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha scaling |

### Command Line Options

```bash
./run_full_pipeline.sh \
    --target google/gemma-3-4b-it \
    --drafter google/gemma-3-270m-it \
    --speculation-depth 4 \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --lora-rank 64
```

---

## 💾 Memory Issues?

Reduce values via environment:

```bash
BATCH_SIZE=2 ./run_full_pipeline.sh              # Reduce from 4 to 2
FEAT_MAX_LENGTH=1024 ./run_full_pipeline.sh      # Reduce from 2048 to 1024
SPECULATION_DEPTH=2 ./run_full_pipeline.sh       # Reduce from 4 to 2
```

---

## 📊 Monitoring Training

### Live Logs
```bash
tail -f logs/training_*.log
```

### Checkpoint Logs
```bash
ls checkpoints/logs/
cat checkpoints/logs/*/training.log
```

### TensorBoard
```bash
tensorboard --logdir checkpoints/logs/
```

---

## 🔍 Common Issues

| Issue | Solution |
|-------|----------|
| `SIGKILL` after 2 steps | Memory cleanup enabled - check GPU memory |
| Slow data loading | Normal for lazy loading, ~5 min per batch |
| NCCL timeout | Set `NCCL_TIMEOUT=7200` |
| Port already in use | Run `fuser -k 29500/tcp` |

---

## ✅ What Works

- ✨ Multi-node DDP training
- ✨ DeepSpeed ZeRO-3 support
- ✨ Memory cleanup to prevent OOM
- ✨ Curriculum learning (MTP heads 1→K)
- ✨ Gradient checkpointing
- ✨ Mixed precision (TF32/BF16)

---

## 🎯 Skipping Stages

```bash
# Skip data generation (use existing data)
./run_full_pipeline.sh --skip-data-gen

# Skip feature extraction (use existing features)
./run_full_pipeline.sh --skip-feature-extraction

# Train only (skip data gen and features)
./run_full_pipeline.sh --skip-data-gen --skip-feature-extraction

# Use custom JSONL input
./run_full_pipeline.sh --input-jsonl data/my_data.jsonl
```

---

## 📁 Output Locations

| Output | Location |
|--------|----------|
| Checkpoints | `checkpoints/best_model/` |
| Logs | `logs/training_*.log` |
| Features | `data/features/` |
| Plots | `plot_scripts/plots/` |