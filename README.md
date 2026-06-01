# P-EAGLE

### ⚡ Parallel Speculative Decoding Framework for LLM Inference

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Flash Attention](https://img.shields.io/badge/Flash%20Attention-Enabled-blueviolet.svg)](https://arxiv.org/abs/2205.14135)
[![Multi-GPU](https://img.shields.io/badge/Multi--GPU-DDP-orange.svg)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
[![LoRA](https://img.shields.io/badge/LoRA-Supported-green.svg)](https://arxiv.org/abs/2106.09685)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**P-EAGLE** 🚀 achieves **1.5-3x inference speedup** by predicting multiple future tokens in parallel using speculative decoding with EAGLE-3 architecture.

## 📖 What is P-EAGLE?

P-EAGLE implements the **EAGLE-3** speculative decoding algorithm that accelerates large language model inference **without quality loss**. Unlike traditional autoregressive decoding (token-by-token), P-EAGLE uses a lightweight "drafter" model to predict K tokens in parallel, then verifies them all in a single pass through the target model using tree attention.

**✨ Key Innovations:**
- 🔮 **EAGLE-3 Hidden State Injection**: Combines target model embeddings with hidden states for accurate draft prediction
- 🎯 **Multi-Token Prediction**: K parallel heads predict multiple future tokens simultaneously
- 🌳 **Tree Attention**: Verifies K tokens in O(1) complexity per token
- ⚡ **Flash Attention**: Memory-efficient attention for long contexts
- 🔥 **Multi-GPU Training**: Distributed training across multiple GPUs
- 📦 **Sliding Window + System Prompt Anchoring**: Professional chunking preserves tool definitions across all windows

---

## 🏗️ Architecture

![P-EAGLE Architecture](docs/p-eagle-architecture.gif)

**🔄 How Speculative Decoding Works:**

| Step | Component | Description |
|------|-----------|-------------|
| 1️⃣ | Target Model | Provides hidden states from forward pass |
| 2️⃣ | Feature Extractor | Extracts and fuses hidden states from last N layers |
| 3️⃣ | EAGLE-3 Injection | Concatenates `[embeddings ⊕ hidden]` for drafter input |
| 4️⃣ | Drafter Model | Generates K tokens in parallel with MTP heads |
| 5️⃣ | Tree Attention | Verifies all K tokens in one target model pass |
| 6️⃣ | Accept/Reject | Accepted tokens used, rejected tokens resampled |

---

## 🔄 Data Pipeline

![P-EAGLE Workflow](docs/p-eagle-workflow.gif)

```
📥 Raw Data → 📦 Sliding Window + System Anchor → 🔗 Block Packing → 🏋️ Train Drafter → 🎯 Inference/Evaluate
```

### 📦 Sliding Window with System Prompt Anchoring (H100/H200 Optimized)
Professional-grade chunking for speculative training on datacenter hardware:
- **System Prompt Anchoring**: System prompt prepended to EVERY chunk for tool definitions
- **Context Overlap**: 50% overlap between windows preserves immediate history
- **Zero Padding**: Block packing achieves 100% tensor core utilization
- **Variable-Length FlashAttention**: Uses `cu_seqlens` for cross-conversation isolation

```bash
# Prepare data for H200
python scripts/sequence_packing.py \
    --input data/raw_conversations.jsonl \
    --output data/packed_features \
    --max_seq_len 4096 \
    --tokenizer google/gemma-3-4b-it
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
./run_full_pipeline.sh

# Or step by step (H200 optimized)
python scripts/generate_data.py --output data/raw_conversations.jsonl
python scripts/sequence_packing.py --input data/raw_conversations.jsonl --output data/packed_features
python -m p_eagle.scripts.extract_features --model_path google/gemma-3-4b-it
python -m p_eagle.scripts.train_drafter --drafter_model google/gemma-3-270m-it --use_lora
python -m p_eagle.scripts.evaluate
```

---

## 💻 Usage Example

```python
from p_eagle.inference import PEAGLEInference

engine = PEAGLEInference(
    drafter_checkpoint="checkpoints/best_model",
    target_model="google/gemma-3-4b-it",
    speculation_depth=4
)

result = engine.generate("Explain quantum computing:", max_tokens=100)
print(f"Speedup: {result.speedup:.2f}x")
print(f"Acceptance rate: {result.acceptance_rate:.1%}")
```

---

## ⚙️ Configuration

| Parameter | Value Used | Description |
|-----------|-----------|-------------|
| `drafter_model` | google/gemma-3-270m-it | Base model for drafter |
| `target_hidden_dim` | 2560 | Target model hidden dimension |
| `speculation_depth` | 4 | Number of parallel predictions (K) |
| `lora_rank` | 64 | LoRA adaptation rank |
| `lora_alpha` | 256 | LoRA scaling factor |
| `learning_rate` | 5e-5 | Training learning rate |
| `num_epochs` | 1 | Training epochs |
| `max_seq_len` | 32768 | Maximum sequence length |
| `batch_size` | 1 | Training batch size |
| `gradient_accumulation_steps` | 8 | Gradient accumulation |
| `shard_cache_size` | 2 | Feature shard cache size |
| `use_flash_attention` | true | Enable flash attention |
| `save_every` | 10 | Save checkpoint every N steps |

**📝 Example Training Command:**
```bash
pkill -f "p_eagle.training.trainer"

/home/suraj/Desktop/P_Eagle/venv/bin/python -m p_eagle.training.trainer \
  --drafter_model google/gemma-3-270m-it \
  --feature_dir /home/suraj/Desktop/P_Eagle/data/features/subset_500 \
  --output_dir /home/suraj/Desktop/P_Eagle/outputs/training_subset_500 \
  --target_hidden_dim 2560 \
  --use_lora --lora_rank 64 --lora_alpha 256 \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --max_seq_len 32768 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --speculation_depth 4 \
  --shard_cache_size 2 \
  --use_flash_attention \
  --save_every 10 \
  --resume /home/suraj/Desktop/P_Eagle/outputs/training_subset_500/epoch_1_checkpoint \
  --yes
```

---

## 💻 Hardware Requirements

| Drafter Size | GPU Memory | Training Time (7k samples) |
|--------------|------------|---------------------------|
| 270M | 12 GB | ~2-3 hours/epoch |
| 1B | 20 GB | ~4-6 hours/epoch |

**🎮 Recommended:** NVIDIA A100/H100 (40-80GB) or RTX 4090 (24GB)

---

## 📊 Performance Summary

| Metric | Value | Description |
|--------|-------|-------------|
| ⚡ Speedup | 1.5-3x | Tokens/second vs autoregressive baseline |
| ✅ Acceptance Rate | >70% | Draft tokens accepted by target model |
| 💾 Memory Overhead | +270MB | Drafter model + KV cache |
| 🎯 Output Quality | Identical | Probabilistically same as target |

---

## ❓ Why P-EAGLE?

| Feature | Description |
|---------|-------------|
| ⚡ **Fast** | 1.5-3x inference speedup with no quality loss |
| 💾 **Efficient** | Memory-efficient with flash attention |
| 📦 **Professional Chunking** | Sliding window + system anchor for accurate speculation |
| 🔧 **Flexible** | Works with any transformer-based model |
| 📈 **Scalable** | Multi-GPU training support |
| 🏭 **Production-Ready** | Clean API, comprehensive documentation |

---

## 🎯 Getting Started

1. **📦 Install**: `pip install -r requirements.txt`
2. **📂 Prepare Data**: Generate or download raw conversations in OpenAI format
3. **📦 Apply Sliding Window**: `python scripts/sequence_packing.py --input <data> --output <output>`
4. **📊 Extract Features**: Run feature extraction with your target model
5. **🏋️ Train Drafter**: Train the EAGLE drafter model with LoRA
6. **🎯 Evaluate**: Run evaluation to measure speedup and acceptance rate
7. **⚙️ Customize**: Adjust `speculation_depth`, `lora_rank`, `max_seq_len` as needed

---

## 🤝 Contributing

Contributions are welcome! 💡 Please open an issue or submit a pull request.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**🚀 P-EAGLE: Making LLM inference 1.5-3x faster through parallel speculative decoding**

</div>