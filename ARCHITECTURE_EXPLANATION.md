# P-EAGLE Architecture Deep Dive

## How It Actually Works

### Training Flow
```
1. Target Model (e.g., Gemma-7B) processes text
   └── Outputs hidden states from layers [7, 14, 21]
   └── These are fused into single vector per token

2. Drafter Model (e.g., Gemma-2B) is trained to predict:
   └── Input: Token IDs (using compatible tokenizer)
   └── Output: Hidden state vectors matching target's fused hidden states

3. Loss Computation:
   └── OLD: MSE between predicted and target hidden states
   └── NEW: KL divergence between token distributions
       ├── Pass predicted hidden through TARGET's lm_head → logits_pred
       ├── Pass target hidden through TARGET's lm_head → logits_target
       └── Minimize KL(logits_pred || logits_target)
```

### Inference Flow
```
1. Drafter sees current token sequence
   └── Generates K parallel hidden state predictions (h_{t+1}, ..., h_{t+K})

2. Convert hidden states to tokens:
   └── Pass each h_{t+i} through TARGET's lm_head
   └── Get token probabilities → sample token

3. Target verifies K drafts in parallel:
   └── Target model sees [context] + [draft_token_1, ..., draft_token_K]
   └── Outputs K logits (one per position)
   └── Accept tokens where target agrees with draft

4. Keep accepted tokens, reject + resample from target for rest
```

## Critical Requirements

### Vocabulary Compatibility
- **Training**: Drafter and target must use COMPATIBLE tokenizers
- **Inference**: Drafter outputs hidden states → target's lm_head → target tokens
- **Key Point**: Token IDs flow through drafter, but tokens come from target's lm_head

### Compatible Model Pairs
| Target | Drafter | Vocab | Works? |
|--------|---------|-------|--------|
| Gemma-7B | Gemma-2B | Same ✅ | Yes |
| Qwen-7B | Qwen-1.5B | Same ✅ | Yes |
| GLM-5.1 | GLM-1.5B | Same ✅ | Yes |
| Gemma-7B | Qwen-1.5B | Different ❌ | No (vocab mismatch) |

## Why Previous Training Failed

### The Problem
Training used MSE loss on hidden state vectors:
```python
loss = MSE(pred_hidden, target_hidden)
```

**BUT**: Two vectors can be very close in MSE but produce completely
different argmax tokens through lm_head!

Example:
- `pred_hidden` → lm_head → token_id=45 ("hello")
- `target_hidden` → lm_head → token_id=12345 (random garbage)
- MSE between vectors: 0.001 (very small)
- Token match: NO ❌

### The Fix
Use KL divergence on token distributions:
```python
pred_logits = target_lm_head(pred_hidden)
target_logits = target_lm_head(target_hidden)
loss = KL(pred_logits, target_logits)
```

This ensures predicted hidden states produce the SAME token distribution
as target hidden states when passed through the target's lm_head.

## For GLM Family Models

Since GLM-5.1 and GLM-1.5B share the same vocabulary:

1. **Feature Extraction**: Use GLM-5.1 to extract hidden states
   ```bash
   python extract_features --model_path THUDM/glm-5.1 --tokenizer_path THUDM/glm-1.5b
   ```

2. **Training**: Train GLM-1.5B to predict GLM-5.1's hidden states
   ```bash
   python train_drafter --drafter_model THUDM/glm-1.5b --target_hidden_dim 4096
   ```

3. **Inference**: GLM-1.5B hidden → GLM-5.1 lm_head → GLM tokens ✓

## Implementation Status

✅ Fixed: Token-level loss during training (KL divergence)
✅ Fixed: Token alignment with target's lm_head
✅ Fixed: Vocab compatibility checking
⚠️  Note: Target lm_head is created as placeholder during training
   - In production, load actual target model's lm_head weights
   - Or train with target model accessible (slower but more accurate)
