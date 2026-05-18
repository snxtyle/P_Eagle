# P-EAGLE Architecture Deep Dive

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           P-EAGLE SYSTEM                                 │
│                    Parallel Speculative Decoding                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │     DATA       │     │   TRAINING    │     │   INFERENCE  │
    │ PREPARATION   │     │    PIPELINE   │     │    ENGINE     │
    └───────────────┘     └───────────────┘     └───────────────┘
```

## Training Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │   Raw Data  │ ───► │ Feature Extract │ ───► │   Train Drafter │
    │ (JSONL/Text)│      │   (Target Model)│      │   (EAGLE-3)     │
    └─────────────┘      └─────────────────┘      └─────────────────┘
                              │                           │
                              ▼                           ▼
                    ┌─────────────────┐         ┌─────────────────┐
                    │ Hidden States   │         │ Trained Checkpoint │
                    │ (layers -1,-2,-3)│         │ + LoRA weights   │
                    └─────────────────┘         └─────────────────┘
```

### Step-by-Step Training Flow

```
Step 1: Data Preparation
────────────────────────
    Input Text ──► Tokenize ──► Input IDs ──► Feature Extraction

Step 2: Feature Extraction (Target Model)
────────────────────────
    Input IDs ──► Target Model (frozen) ──► Hidden States
                  ├── Layer -1 (final)
                  ├── Layer -2 (middle)
                  └── Layer -3 (early)

Step 3: Feature Fusion
────────────────────────
    [h_{-1}, h_{-2}, h_{-3}] ──► Mean/Concat ──► Fused Hidden State

Step 4: Drafter Training (EAGLE-3)
────────────────────────
    Drafter Input:
    ┌─────────────────────────────────────────────────┐
    │ [Token Embeddings] + [Target Hidden States]     │
    │        640-dim           +       2560-dim        │
    │              =            3200-dim               │
    └─────────────────────────────────────────────────┘

    First Layer Modification:
    ┌─────────────────────────────────────────────────┐
    │ Original:  input_dim = 640                       │
    │ Modified:  input_dim = 1280 (2x for concat)       │
    │            └── Separate LayerNorms for            │
    │                embeddings and hidden states      │
    └─────────────────────────────────────────────────┘

Step 5: Loss Computation (KL Divergence)
────────────────────────
    Pred Hidden ──► Target lm_head ──► Pred Logits
                                            │
    Target Hidden ──► Target lm_head ──► Target Logits
                                            │
                    KL Divergence ◄────────┘

Step 6: Backpropagation
────────────────────────
    Loss ──► Update LoRA params + MTP heads + Projection
```

## Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │  User Input │ ───► │   Drafter       │ ───► │    Target       │
    │   (Prompt)  │      │  Speculation    │      │  Verification   │
    └─────────────┘      └─────────────────┘      └─────────────────┘
                                │                           │
                                ▼                           ▼
                    ┌─────────────────┐         ┌─────────────────┐
                    │ K draft tokens  │         │ Accepted tokens │
                    │ (parallel)      │         │ + Re-sampled    │
                    └─────────────────┘         └─────────────────┘
```

### Step-by-Step Inference Flow

```
Step 1: Initial Context
────────────────────────
    Prompt ──► Tokenize ──► Drafter processes ──► Current token hidden

Step 2: Speculation (Drafter generates K drafts)
────────────────────────
    Current hidden ──► MTP Head 1 ──► hidden_1 ──► sample ──► token_1
    Current hidden ──► MTP Head 2 ──► hidden_2 ──► sample ──► token_2
    Current hidden ──► MTP Head 3 ──► hidden_3 ──► sample ──► token_3
    Current hidden ──► MTP Head 4 ──► hidden_4 ──► sample ──► token_4
         │                │              │            │
         └────────────────┴──────────────┴────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  K Draft Tokens:   │
                    │  [t₁, t₂, t₃, t₄]  │
                    └───────────────────┘

Step 3: Tree Attention Setup
────────────────────────
    Draft Tree Structure:
                    root
                   / | \
                  t₁ t₂ t₃
                  |
                 t₄

    Attention Mask:
    ┌────────────────────────────┐
    │ Context can see all        │
    │ t₁ can see context + t₁    │
    │ t₂ can see context + t₁    │
    │ t₃ can see context + t₁   │
    │ t₄ can see context + t₁   │
    └────────────────────────────┘

Step 4: Parallel Verification (Target Model)
────────────────────────
    [Context + Draft Tree] ──► Target Model ──► K verification logits

Step 5: Token Acceptance
────────────────────────
    For each draft token:
    ├── Draft token probability > threshold? → ACCEPT
    └── Draft token probability < threshold? → REJECT + RESAMPLE

    Example:
    ─────────────────────────────────────────────────────
    Draft: [the, cat, sat, on]
    Target Logits at each position:
    ├── P("the") = 0.95 > 0.8 → ACCEPT ✓
    ├── P("cat") = 0.85 > 0.8 → ACCEPT ✓
    ├── P("mat") = 0.3  < 0.8 → REJECT ✗
    └── P("on") = 0.1   < 0.8 → REJECT ✗

    Final Output: "the cat [resample] [resample]"

Step 6: Output Generation
────────────────────────
    Accepted tokens + new tokens ──► Decode ──► Output text
```

## EAGLE-3 Architecture Details

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EAGLE-3 DRAFTER MODEL                          │
└─────────────────────────────────────────────────────────────────────────┘

Input Layer (Modified)
────────────────────────
    ┌──────────────────┬──────────────────┐
    │ Token Embeddings │ Target Hidden   │
    │    (640-dim)     │   (2560-dim)    │
    └────────┬─────────┴────────┬────────┘
             │                    │
        LayerNorm            LayerNorm
             │                    │
             └──────────┬─────────┘
                        │
                   Concatenate
                        │
                        ▼
                   3200-dim
                        │
                        ▼
              ┌─────────────────┐
              │  Q, K, V Proj   │ (LoRA adapted)
              │  O Proj         │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  FFN Layers    │ (LoRA adapted)
              └─────────────────┘
                        │
                        ▼
                 Hidden States
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
    ┌───────────────┐     ┌───────────────┐
    │   Standard     │     │  MTP Heads    │
    │  Attention     │     │  (K heads)    │
    └───────────────┘     └───────────────┘

Multi-Token Prediction (MTP) Heads
────────────────────────
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  Hidden ──► MTP_1 ──► h₁ ──► lm_head ──► t₁ │
    │  Hidden ──► MTP_2 ──► h₂ ──► lm_head ──► t₂ │
    │  Hidden ──► MTP_3 ──► h₃ ──► lm_head ──► t₃ │
    │  Hidden ──► MTP_4 ──► h₄ ──► lm_head ──► t₄ │
    │                                             │
    │  (All use TARGET's lm_head for vocab)        │
    └─────────────────────────────────────────────┘
```

## Vocabulary Compatibility Requirement

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VOCABULARY COMPATIBILITY                             │
└─────────────────────────────────────────────────────────────────────────┘

    ✅ COMPATIBLE (Same Tokenizer)          ❌ INCOMPATIBLE (Different Tokenizers)
    ─────────────────────────────            ──────────────────────────────
    Gemma-7B + Gemma-2B                    Gemma-7B + Qwen-1.5B
    Qwen-7B + Qwen-1.5B                     Llama-7B + Mistral-7B
    GLM-5.1 + GLM-1.5B

Training Process:
────────────────────────
    Target Hidden ──► Target lm_head ──► Logits ──► Loss
    Draft Hidden  ──► Target lm_head ──► Logits ◄──┘

    Key: Draft uses TARGET's lm_head during training!
```

## Why KL Divergence (Not MSE)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOSS FUNCTION COMPARISON                        │
└─────────────────────────────────────────────────────────────────────────┘

    MSE on Hidden States (WRONG ❌)
    ─────────────────────────────
    pred_hidden ≈ target_hidden (MSE = 0.001)
         │                    │
         ▼                    ▼
    lm_head            lm_head
         │                    │
         ▼                    ▼
    "hello"            "xyz123"  (WRONG TOKENS!)

    KL Divergence on Distributions (CORRECT ✅)
    ──────────────────────────────────────────
    pred_logits ≈ target_logits (KL = 0.01)
         │                    │
         ▼                    ▼
    argmax                   argmax
         │                    │
         ▼                    ▼
    "hello" ◄──────────────► "hello"  (SAME TOKENS!)
```

## Data Flow Summary

```
TRAINING:
────────────────────────────────────────────────────────────────────────
Raw Text ─► Tokenize ─► Target Model ─► Hidden States ─► Fuse ─► Drafter
                                                                    │
                                                                    ▼
                                                              Loss (KL)
                                                                    │
                                                                    ▼
                                                              Backprop

INFERENCE:
────────────────────────────────────────────────────────────────────────
Prompt ─► Tokenize ─► Drafter ─► Draft Tokens ─► Target Verifies
                                                         │
                                                         ▼
                                                   Accept/Reject
                                                         │
                                                         ▼
                                                   Output Text
```

## File-to-Module Mapping

```
┌─────────────────────┐    ┌─────────────────────┐
│    TRAINING         │    │     INFERENCE       │
├─────────────────────┤    ├─────────────────────┤
│ scripts/            │    │ scripts/            │
│  └─ train_drafter   │    │  └─ run_inference   │
│                     │    │                     │
│ training/           │    │ inference/          │
│  ├─ trainer.py      │    │  └─ inference_engine│
│  └─ feature_extract │    │                     │
│                     │    │                     │
│ models/             │    │ models/             │
│  ├─ peagle_drafter │    │  ├─ tree_attention  │
│  └─ flash_attention│    │  └─ peagle_drafter  │
└─────────────────────┘    └─────────────────────┘
```