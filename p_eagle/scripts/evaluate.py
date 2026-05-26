#!/usr/bin/env python3
"""
P-EAGLE Evaluation Script - Production-Grade Speculative Decoding Evaluation

Complete metrics framework for evaluating P-EAGLE speculative decoding:

1. PRIMARY PERFORMANCE METRICS (The Efficiency Engine)
   - Mean Acceptance Length (MAL): Average tokens accepted per verification pass
   - Tokens Per Second (TPS): Generation speed
   - Wall-Clock Speedup: Speedup factor vs baseline

2. MICRO-ARCHITECTURAL METRICS (The Structural Health)
   - Per-Head Acceptance Rate: Individual acceptance per MTP head position
   - Target Pass Efficiency: Tokens generated per verification pass
   - Overall Acceptance Rate: Total accepted / total drafted

3. QUALITY & GUARDRAIL METRICS
   - Perplexity: Language model quality
   - Text Equivalence: Output similarity to baseline
   - Token Distribution Parity: Distribution match with target model
"""

import argparse
import json
import time
import sys
import os
import math
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch._inductor.config.triton.cudagraphs = False
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"


# =============================================================================
# METRIC CALCULATION FUNCTIONS WITH EXAMPLES
# =============================================================================

def calculate_mal(acceptance_counts: List[int]) -> float:
    """
    Calculate Mean Acceptance Length (MAL).

    MAL = sum(accepted_tokens) / num_verification_passes

    Example:
        acceptance_counts = [3, 1, 4, 2, 3]  # tokens accepted at each step
        MAL = (3 + 1 + 4 + 2 + 3) / 5 = 2.6

    Target: MAL >= 2.0 for K=4 on technical text
    """
    if not acceptance_counts:
        return 0.0
    return np.mean(acceptance_counts)


def calculate_tps(tokens_generated: int, elapsed_time: float) -> float:
    """
    Calculate Tokens Per Second (TPS).

    TPS = tokens_generated / elapsed_time

    Example:
        tokens_generated = 100
        elapsed_time = 5.2 seconds
        TPS = 100 / 5.2 = 19.23 tokens/second
    """
    if elapsed_time <= 0:
        return 0.0
    return tokens_generated / elapsed_time


def calculate_speedup(drafter_tps: float, baseline_tps: float) -> float:
    """
    Calculate speedup factor.

    Speedup = drafter_tps / baseline_tps

    Example:
        baseline_tps = 8.5
        drafter_tps = 18.2
        Speedup = 18.2 / 8.5 = 2.14x

    Target: >= 1.8x to 2.5x for production
    """
    if baseline_tps <= 0:
        return 0.0
    return drafter_tps / baseline_tps


def calculate_per_head_acceptance(
    acceptance_counts: List[int],
    speculation_depth: int,
    total_samples: int
) -> Dict[int, float]:
    """
    Calculate per-head acceptance rate.

    For each head position h, calculate what percentage of samples
    had at least h accepted tokens (meaning head h was accepted).

    Example:
        speculation_depth = 4
        acceptance_counts = [3, 1, 4, 2, 3, 0, 4, 1]

        For head 1: samples with accepted >= 1 = 7/8 = 87.5%
        For head 2: samples with accepted >= 2 = 5/8 = 62.5%
        For head 3: samples with accepted >= 3 = 3/8 = 37.5%
        For head 4: samples with accepted >= 4 = 2/8 = 25.0%

    Target decay curve:
        Head 1: 75-85%
        Head 2: 55-65%
        Head 3: 40-48%
        Head 4: 25-35%
    """
    per_head = {}
    for h in range(1, speculation_depth + 2):  # +2 for beyond K and >0
        accepted_count = sum(1 for ac in acceptance_counts if ac >= h)
        per_head[h] = (accepted_count / total_samples * 100) if total_samples > 0 else 0.0
    return per_head


def calculate_target_pass_efficiency(
    total_tokens: int,
    num_verification_passes: int
) -> float:
    """
    Calculate Target Pass Efficiency.

    Efficiency = total_tokens / num_verification_passes

    This measures how many tokens we generate per call to the target model.
    Higher is better - means we're amortizing the target model cost.

    Example:
        total_tokens = 100
        num_verification_passes = 45
        Efficiency = 100 / 45 = 2.22 tokens/pass

    Target: >= 2.0 tokens/pass is good
    """
    if num_verification_passes <= 0:
        return 0.0
    return total_tokens / num_verification_passes


def calculate_perplexity(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """
    Calculate perplexity of the model.

    Perplexity = exp(average_cross_entropy_loss)

    Lower is better. 1.0 means perfect prediction.

    Example:
        cross_entropy_loss = 2.3
        Perplexity = exp(2.3) = 9.97
    """
    if logits.shape[0] == 0 or target_ids.shape[0] == 0:
        return float('inf')
    try:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction='mean'
        )
        return float(torch.exp(loss))
    except:
        return float('inf')


def calculate_token_distribution_parity(
    drafter_probs: torch.Tensor,
    target_probs: torch.Tensor,
    k: int = 10
) -> Dict[str, float]:
    """
    Calculate token distribution similarity between drafter and target.

    Uses KL divergence and top-k token overlap.

    KL_div = sum(p * log(p / q)) for all tokens
    Top-k overlap = |top_k_drafter ∩ top_k_target| / k

    Example:
        drafter top-5: [the, a, is, cat, sat]
        target top-5: [the, a, is, cat, dog]
        Overlap = 4/5 = 80%

    Target: >= 70% top-5 overlap for good drafter
    """
    # Clip to avoid log(0)
    drafter_probs = torch.clamp(drafter_probs, min=1e-10)
    target_probs = torch.clamp(target_probs, min=1e-10)

    # KL divergence (drafter || target)
    kl_div = torch.sum(drafter_probs * torch.log(drafter_probs / target_probs)).item()

    # Top-k overlap
    drafter_top_k = torch.topk(drafter_probs, k).indices.tolist()
    target_top_k = torch.topk(target_probs, k).indices.tolist()
    top_k_overlap = len(set(drafter_top_k) & set(target_top_k)) / k * 100

    return {
        'kl_divergence': kl_div,
        'top_k_overlap_pct': top_k_overlap,
        'drafter_top_k': drafter_top_k,
        'target_top_k': target_top_k
    }


def calculate_text_equivalence(
    baseline_tokens: List[int],
    drafter_tokens: List[int],
    tokenizer: Any
) -> Dict[str, Any]:
    """
    Calculate text equivalence between baseline and drafter outputs.

    Measures how similar the outputs are.

    Example:
        baseline: "The cat sat on the mat"
        drafter: "The cat sat on the floor"

        Exact match: False
        Token match rate: 5/6 = 83.3%
        Common prefix length: 4 tokens
    """
    # Exact match
    exact_match = (baseline_tokens == drafter_tokens)

    # Token match rate (intersection / union)
    baseline_set = set(baseline_tokens)
    drafter_set = set(drafter_tokens)
    intersection = len(baseline_set & drafter_set)
    union = len(baseline_set | drafter_set)
    jaccard_similarity = intersection / union if union > 0 else 0

    # Common prefix length
    common_prefix = 0
    for b, d in zip(baseline_tokens, drafter_tokens):
        if b == d:
            common_prefix += 1
        else:
            break

    # Decode for readability
    baseline_text = tokenizer.decode(baseline_tokens) if baseline_tokens else ""
    drafter_text = tokenizer.decode(drafter_tokens) if drafter_tokens else ""

    return {
        'exact_match': exact_match,
        'jaccard_similarity': jaccard_similarity * 100,  # as percentage
        'common_prefix_tokens': common_prefix,
        'baseline_length': len(baseline_tokens),
        'drafter_length': len(drafter_tokens),
        'length_match_pct': min(len(baseline_tokens), len(drafter_tokens)) / max(len(baseline_tokens), len(drafter_tokens), 1) * 100,
        'baseline_text': baseline_text[:100] + "..." if len(baseline_text) > 100 else baseline_text,
        'drafter_text': drafter_text[:100] + "..." if len(drafter_text) > 100 else drafter_text,
    }


def evaluate_baseline(target_model_name: str, prompts: List[str], max_tokens: int = 100,
                      temperature: float = 0.7, top_p: float = 0.9,
                      quantization: str = "none") -> Dict:
    """Evaluate raw target model baseline with comprehensive metrics."""
    print("\n" + "="*70)
    print("  BASELINE: Raw Target Model")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {target_model_name} (quantization={quantization})...")
    load_kwargs = {"device_map": "auto"}
    quant_config = _get_quant_config(quantization)
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(target_model_name, **load_kwargs)
    model.eval()

    results = []
    total_time = 0
    total_tokens = 0
    all_perplexities = []
    all_generated_tokens = []  # For text equivalence comparison

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] Generating...", end=" ", flush=True)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        original_length = input_ids.shape[1]

        start = time.time()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )
        elapsed = time.time() - start

        generated_ids = output[0, original_length:].tolist()
        tokens_generated = len(generated_ids)
        all_generated_tokens.append(generated_ids)

        # Calculate perplexity of generated tokens
        with torch.no_grad():
            full_output = model(output)
            logits = full_output.logits[0, original_length-1:-1]
            perplexity = calculate_perplexity(logits, output[0, original_length:])

        tps = calculate_tps(tokens_generated, elapsed)

        results.append({
            "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
            "tokens": tokens_generated,
            "time": elapsed,
            "tps": tps,
            "perplexity": perplexity,
            "generated_text": tokenizer.decode(generated_ids)[:100]
        })

        total_time += elapsed
        total_tokens += tokens_generated
        all_perplexities.append(perplexity)
        print(f"{tokens_generated} tokens, {elapsed:.2f}s, {tps:.1f} tps, ppl={perplexity:.2f}")

    mean_tps = calculate_tps(total_tokens, total_time)

    return {
        "model": target_model_name,
        "total_samples": len(prompts),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "mean_tps": mean_tps,
        "mean_perplexity": np.mean(all_perplexities) if all_perplexities else float('inf'),
        "min_perplexity": np.min(all_perplexities) if all_perplexities else float('inf'),
        "max_perplexity": np.max(all_perplexities) if all_perplexities else float('inf'),
        "std_perplexity": np.std(all_perplexities) if all_perplexities else 0,
        "samples": results,
        "_raw_tokens": all_generated_tokens  # For comparison with drafter
    }


def _get_quant_config(quantization: str):
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    return None


def evaluate_speculative(drafter_checkpoint: str, target_model_name: str,
                         prompts: List[str], max_tokens: int = 100,
                         temperature: float = 0.7, top_p: float = 0.9,
                         quantization: str = "none") -> Dict:
    """Evaluate with TRUE speculative decoding (no teacher forcing).

    Calculates comprehensive metrics including:
    - Mean Acceptance Length (MAL)
    - Per-Head Acceptance Rate
    - Target Pass Efficiency
    - Tokens Per Second (TPS)
    - Perplexity
    """
    from p_eagle.models.peagle_drafter import EagleDrafterModel

    print("\n" + "="*70)
    print("  P-EAGLE: Speculative Decoding")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load target model for verification (use same quantization as feature extraction)
    print(f"Loading target model: {target_model_name} (quantization={quantization})...")
    load_kwargs = {"device_map": "auto"}
    quant_config = _get_quant_config(quantization)
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, **load_kwargs)
    target_model.eval()

    # Load drafter
    print(f"Loading drafter from {drafter_checkpoint}...")

    # Check if drafter_checkpoint is a HuggingFace model ID or a local checkpoint
    is_hf_model = "/" in drafter_checkpoint and not os.path.exists(drafter_checkpoint)

    if is_hf_model:
        # For base model evaluation: we need to know the dimensions
        # gemma-3-270m-it: 640 hidden dim, gemma-3-4b-it: 2560 hidden dim
        base_drafter_dim = 640
        base_target_dim = 2560

        # Create EagleDrafterModel with untrained weights
        from p_eagle.models.peagle_drafter import EagleDrafterModel
        drafter = EagleDrafterModel(
            base_model_name=drafter_checkpoint,
            target_hidden_dim=base_target_dim,
            speculation_depth=4,  # Use default K=4
            use_lora=False,
            device="cuda",
            use_hidden_injection=True,
            injection_mode="concat"
        )
        print(f"Created untrained drafter model with MTP heads for: {drafter_checkpoint}")
    else:
        # Load trained P-EAGLE checkpoint
        drafter = EagleDrafterModel.load_checkpoint(drafter_checkpoint, device="cuda")

    drafter.eval()

    # Get the target model's lm_head for token generation
    if hasattr(target_model, 'lm_head'):
        target_lm_head = target_model.lm_head
    elif hasattr(target_model, 'model') and hasattr(target_model.model, 'lm_head'):
        target_lm_head = target_model.model.lm_head
    else:
        raise ValueError("Could not find lm_head in target model")

    drafter_output_dim = 640  # gemma-3-270m-it's embedding/output dimension
    target_hidden_dim = 2560  # What the target model expects (from training)

    print(f"Drafter output dim: {drafter_output_dim}")
    print(f"Target hidden dim: {target_hidden_dim}")

    # Load projection weights from drafter if available
    if hasattr(drafter, 'target_hidden_proj'):
        with torch.no_grad():
            src_weight = drafter.target_hidden_proj.weight
            src_bias = drafter.target_hidden_proj.bias
            lm_head_projection = torch.nn.Linear(target_hidden_dim, drafter_output_dim, dtype=torch.bfloat16).to("cuda")
            with torch.no_grad():
                lm_head_projection.weight.copy_(src_weight)
                if src_bias is not None:
                    lm_head_projection.bias.copy_(src_bias)
        print(f"Loaded projection weights: {target_hidden_dim} -> {drafter_output_dim}")
    else:
        print(f"Warning: No projection weights found, using random init")
        lm_head_projection = torch.nn.Linear(target_hidden_dim, drafter_output_dim, dtype=torch.bfloat16).to("cuda")

    speculation_depth = drafter.speculation_depth
    print(f"Speculation depth (K): {speculation_depth}")

    # Initialize accumulators for comprehensive metrics
    results = []
    all_acceptance_by_head = {h: [] for h in range(1, speculation_depth + 2)}
    all_mal = []
    total_drafted = 0
    total_accepted = 0
    total_verification_passes = 0  # For Target Pass Efficiency
    all_perplexities = []
    all_generated_tokens = []  # For text equivalence

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] Speculating...", end=" ", flush=True)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        original_length = input_ids.shape[1]
        generated = input_ids.clone()

        start = time.time()
        acceptance_counts = []
        num_verification_passes = 0

        # Get initial target hidden states for the prompt prefix
        with torch.no_grad():
            init_outputs = target_model(generated, output_hidden_states=True)
            target_hidden = init_outputs.hidden_states[-1]  # [1, seq_len, target_hidden_dim]

        # Speculative decoding loop
        for _ in range(max_tokens):
            if generated.shape[1] >= original_length + max_tokens:
                break

            # Step 1: Drafter generates K draft tokens
            with torch.no_grad():
                drafter_outputs = drafter.forward(
                    input_ids=generated,
                    target_hidden=target_hidden,
                    is_training=False
                )
                mtp_predictions = drafter_outputs["mtp_predictions"]

            # Convert hidden states to K draft tokens
            draft_tokens = []
            for k in range(min(speculation_depth, max_tokens - (generated.shape[1] - original_length))):
                pred_hidden = mtp_predictions[k]
                logits = target_lm_head(pred_hidden[:, -1:, :])
                token_id = torch.argmax(logits, dim=-1).item()
                draft_tokens.append(token_id)

            if not draft_tokens:
                break

            # Step 2: Target verifies ALL drafts in parallel
            draft_tensor = torch.tensor([draft_tokens], device="cuda")
            verify_input = torch.cat([generated, draft_tensor], dim=1)
            num_verification_passes += 1

            with torch.no_grad():
                verify_outputs = target_model(verify_input, output_hidden_states=True)
                verify_logits = verify_outputs.logits[0, generated.shape[1]-1:, :]
                verify_hidden = verify_outputs.hidden_states[-1]  # [1, verify_len, target_hidden_dim]

            # Step 3: Accept/reject tokens
            accepted_count = 0
            for j, draft_token in enumerate(draft_tokens):
                target_token = torch.argmax(verify_logits[j]).item()
                if draft_token == target_token:
                    accepted_count += 1
                    total_accepted += 1
                else:
                    break

            # Record acceptance at each depth level
            for h in range(1, speculation_depth + 2):
                if accepted_count >= h:
                    all_acceptance_by_head[h].append(1)
                else:
                    all_acceptance_by_head[h].append(0)

            # Append accepted tokens
            if accepted_count > 0:
                new_tokens = torch.tensor([[draft_tokens[i] for i in range(accepted_count)]], device="cuda")
                generated = torch.cat([generated, new_tokens], dim=1)
                acceptance_counts.append(accepted_count)
                total_drafted += len(draft_tokens)
            else:
                # All rejected - target generates one token
                fallback_token = torch.argmax(verify_logits[0]).item()
                new_token = torch.tensor([[fallback_token]], device="cuda")
                generated = torch.cat([generated, new_token], dim=1)
                acceptance_counts.append(0)
                total_drafted += len(draft_tokens)

            # Update target_hidden from verification outputs (covers the new confirmed prefix)
            target_hidden = verify_hidden[:, :generated.shape[1], :]

        elapsed = time.time() - start
        tokens_generated = generated.shape[1] - original_length
        generated_ids = generated[0, original_length:].tolist()
        all_generated_tokens.append(generated_ids)

        # Calculate MAL using helper function
        mal = calculate_mal(acceptance_counts)
        all_mal.append(mal)

        # Calculate perplexity
        with torch.no_grad():
            full_output = target_model(generated)
            logits = full_output.logits[0, original_length-1:-1]
            if len(generated_ids) > 0 and logits.shape[0] > 0:
                perplexity = calculate_perplexity(logits, generated[0, original_length:])
            else:
                perplexity = float('inf')
        all_perplexities.append(perplexity)

        tps = calculate_tps(tokens_generated, elapsed)

        # Calculate Target Pass Efficiency for this sample
        sample_efficiency = calculate_target_pass_efficiency(tokens_generated, num_verification_passes)

        results.append({
            "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
            "tokens": tokens_generated,
            "mal": mal,
            "time": elapsed,
            "tps": tps,
            "perplexity": perplexity,
            "accepted": sum(acceptance_counts),
            "drafted": len(acceptance_counts) * speculation_depth,  # Approximate
            "verification_passes": num_verification_passes,
            "target_pass_efficiency": sample_efficiency,
            "generated_text": tokenizer.decode(generated_ids)[:100]
        })

        total_verification_passes += num_verification_passes

        print(f"{tokens_generated} tokens, MAL={mal:.2f}, {tps:.1f} tps, "
              f"eff={sample_efficiency:.2f}, ppl={perplexity:.2f}")

    # Calculate aggregate metrics using helper functions
    mean_tps = calculate_tps(sum(r["tokens"] for r in results), sum(r["time"] for r in results))
    overall_acceptance_rate = (total_accepted / total_drafted * 100) if total_drafted > 0 else 0
    mean_mal = np.mean(all_mal) if all_mal else 0.0
    mean_perplexity = np.mean(all_perplexities) if all_perplexities else float('inf')
    mean_efficiency = calculate_target_pass_efficiency(
        sum(r["tokens"] for r in results),
        total_verification_passes
    )

    # Calculate per-head acceptance rate from per-step acceptance data
    acceptance_by_head_pct = {}
    for head, counts in all_acceptance_by_head.items():
        acceptance_by_head_pct[head] = (sum(counts) / len(counts) * 100) if counts else 0.0

    return {
        "drafter_checkpoint": drafter_checkpoint,
        "target_model": target_model_name,
        "speculation_depth": speculation_depth,
        "total_samples": len(prompts),
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "total_verification_passes": total_verification_passes,
        "overall_acceptance_rate": overall_acceptance_rate,
        "mean_mal": mean_mal,
        "mean_tps": mean_tps,
        "mean_perplexity": mean_perplexity,
        "target_pass_efficiency": mean_efficiency,
        "acceptance_by_head": acceptance_by_head_pct,
        "samples": results,
        "_raw_tokens": all_generated_tokens  # For text equivalence
    }


def main():
    parser = argparse.ArgumentParser(
        description="P-EAGLE Production-Grade Speculative Decoding Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with baseline comparison
  python -m p_eagle.scripts.evaluate \\
      --drafter_checkpoint checkpoints/best_model \\
      --target_model google/gemma-3-4b-it \\
      --max_tokens 150 \\
      --baseline \\
      --output metrics_production_report.json

  # Run drafter-only evaluation
  python -m p_eagle.scripts.evaluate \\
      --drafter_checkpoint checkpoints/best_model \\
      --target_model google/gemma-3-4b-it \\
      --max_tokens 100
        """
    )
    parser.add_argument("--drafter_checkpoint", default="./checkpoints/best_model",
                       help="Path to trained drafter checkpoint")
    parser.add_argument("--target_model", default="google/gemma-3-4b-it",
                       help="Target model (MUST match the one used for feature extraction)")
    parser.add_argument("--test_prompts", default=None,
                       help="JSON file with test prompts")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Max tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--quantization", default="none", choices=["4bit", "8bit", "none"],
                       help="Quantization for target model (must match feature extraction)")
    parser.add_argument("--baseline", action="store_true", default=True,
                       help="Also evaluate baseline (no drafter)")
    parser.add_argument("--output", default="eval_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    # Load tokenizer for text equivalence analysis
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    if args.test_prompts:
        with open(args.test_prompts) as f:
            prompts = json.load(f)
    else:
        prompts = [
            "Explain how payment processing works in 3 steps.",
            "What are common causes of SDK integration failures?",
            "Summarize the key metrics for monitoring payment health.",
            "Describe the difference between synchronous and asynchronous callbacks.",
            "How would you troubleshoot a timeout error in a payment gateway?",
            "Write a short Python function to validate credit card numbers.",
            "What is the difference between idempotent and non-idempotent API calls?",
            "Explain WebSocket connections and their use cases.",
            "How does 2FA improve payment security?",
            "Describe the payment gateway authorization flow.",
        ]

    results = {"config": vars(args)}

    # Evaluate baseline first
    baseline_tokens = []
    if args.baseline:
        print("\n" + "="*70)
        print("STEP 1: Baseline Evaluation")
        print("="*70)
        baseline_results = evaluate_baseline(args.target_model, prompts, args.max_tokens,
                                            args.temperature, args.top_p,
                                            quantization=args.quantization)
        results["baseline"] = baseline_results
        baseline_tokens = baseline_results.get("_raw_tokens", [])
        baseline_tps = baseline_results["mean_tps"]
        baseline_ppl = baseline_results["mean_perplexity"]

    # Evaluate with drafter
    print("\n" + "="*70)
    print("STEP 2: Speculative Decoding Evaluation")
    print("="*70)
    drafter_results = evaluate_speculative(args.drafter_checkpoint, args.target_model,
                                          prompts, args.max_tokens, args.temperature, args.top_p,
                                          quantization=args.quantization)
    results["drafter"] = drafter_results
    drafter_tokens = drafter_results.get("_raw_tokens", [])
    drafter_tps = drafter_results["mean_tps"]
    drafter_ppl = drafter_results["mean_perplexity"]

    # Calculate speedup
    speedup = 1.0
    if args.baseline and baseline_tps > 0:
        speedup = calculate_speedup(drafter_tps, baseline_tps)
        ppl_ratio = drafter_ppl / baseline_ppl if baseline_ppl > 0 else 1.0

        # Calculate text equivalence
        print("\n" + "="*70)
        print("STEP 3: Text Equivalence Analysis")
        print("="*70)

        text_equivalence_results = []
        exact_matches = 0
        jaccard_similarities = []
        common_prefixes = []

        for i, (base_toks, draft_toks) in enumerate(zip(baseline_tokens, drafter_tokens)):
            equiv = calculate_text_equivalence(base_toks, draft_toks, tokenizer)
            text_equivalence_results.append(equiv)
            if equiv['exact_match']:
                exact_matches += 1
            jaccard_similarities.append(equiv['jaccard_similarity'])
            common_prefixes.append(equiv['common_prefix_tokens'])

        results["text_equivalence"] = {
            "exact_match_rate": (exact_matches / len(prompts) * 100) if prompts else 0,
            "mean_jaccard_similarity": np.mean(jaccard_similarities) if jaccard_similarities else 0,
            "mean_common_prefix": np.mean(common_prefixes) if common_prefixes else 0,
            "samples": text_equivalence_results
        }

        print(f"  Exact Match Rate: {exact_matches / len(prompts) * 100:.1f}%")
        print(f"  Mean Jaccard Similarity: {np.mean(jaccard_similarities):.1f}%")
        print(f"  Mean Common Prefix: {np.mean(common_prefixes):.1f} tokens")

        # =================================================================
        # PRODUCTION REPORT
        # =================================================================
        print("\n" + "="*70)
        print("  PRODUCTION EVALUATION REPORT")
        print("="*70)

        print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│                     PRIMARY PERFORMANCE METRICS                       │
├──────────────────────────────────────────────────────────────────────┤
│  Metric                        │ Baseline  │ P-EAGLE   │ Target     │
├───────────────────────────────┼───────────┼───────────┼────────────┤
│  Tokens Per Second (TPS)      │ {baseline_tps:9.2f} │ {drafter_tps:9.2f} │ >= 10.0     │
│  Speedup Factor                │     --    │ {speedup:9.2f}x │ >= 1.8x     │
└───────────────────────────────┴───────────┴───────────┴────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     SPEculation METRICS                               │
├──────────────────────────────────────────────────────────────────────┤
│  Metric                        │ Value      │ Target     │ Status    │
├───────────────────────────────┼────────────┼────────────┼───────────┤
│  Mean Acceptance Length (MAL) │ {drafter_results['mean_mal']:10.2f} │ >= 2.0     │ {'✅' if drafter_results['mean_mal'] >= 2.0 else '❌'}   │
│  Overall Acceptance Rate       │ {drafter_results['overall_acceptance_rate']:10.1f}% │ >= 60%     │ {'✅' if drafter_results['overall_acceptance_rate'] >= 60 else '❌'}   │
│  Target Pass Efficiency        │ {drafter_results['target_pass_efficiency']:10.2f} │ >= 2.0     │ {'✅' if drafter_results['target_pass_efficiency'] >= 2.0 else '❌'}   │
└───────────────────────────────┴────────────┴────────────┴───────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     PER-HEAD ACCEPTANCE RATES                        │
├──────────────────────────────────────────────────────────────────────┤""")

        # Print per-head acceptance with targets
        head_targets = {1: (75, 85), 2: (55, 65), 3: (40, 48), 4: (25, 35)}
        for head, rate in sorted(drafter_results['acceptance_by_head'].items()):
            if head in head_targets:
                target_low, target_high = head_targets[head]
                status = '✅' if target_low <= rate <= target_high else '⚠️'
                print(f"│  Head {head}: {rate:5.1f}% (target: {target_low}-{target_high}%)    {status}  │")

        print(f"""└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     QUALITY METRICS                                  │
├──────────────────────────────────────────────────────────────────────┤
│  Metric                        │ Baseline  │ P-EAGLE   │ Delta      │
├───────────────────────────────┼───────────┼───────────┼────────────┤
│  Perplexity                    │ {baseline_ppl:9.2f} │ {drafter_ppl:9.2f} │ {drafter_ppl - baseline_ppl:+9.2f}   │
└───────────────────────────────┴───────────┴───────────┴────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     TEXT EQUIVALENCE                                 │
├──────────────────────────────────────────────────────────────────────┤
│  Exact Match Rate:         {results['text_equivalence']['exact_match_rate']:5.1f}%                        │
│  Mean Jaccard Similarity:  {results['text_equivalence']['mean_jaccard_similarity']:5.1f}%                        │
│  Mean Common Prefix:       {results['text_equivalence']['mean_common_prefix']:.1f} tokens                        │
└──────────────────────────────────────────────────────────────────────┘
""")

        # Final verdict
        print("="*70)
        print("  VERDICT")
        print("="*70)

        all_good = (
            speedup >= 1.5 and
            drafter_results['mean_mal'] >= 1.5 and
            drafter_results['overall_acceptance_rate'] >= 50
        )

        if all_good:
            print("  ✅ P-EAGLE is working effectively!")
            print(f"     - Speedup: {speedup:.2f}x")
            print(f"     - MAL: {drafter_results['mean_mal']:.2f}")
            print(f"     - Acceptance: {drafter_results['overall_acceptance_rate']:.1f}%")
        else:
            print("  ⚠️  P-EAGLE needs optimization")
            print(f"     - Speedup: {speedup:.2f}x (need >= 1.5x)")
            print(f"     - MAL: {drafter_results['mean_mal']:.2f} (need >= 1.5)")
            print(f"     - Acceptance: {drafter_results['overall_acceptance_rate']:.1f}% (need >= 50%)")

        print("="*70)

    else:
        # Drafter-only mode
        print("\n" + "="*70)
        print("  DRAFTER-ONLY EVALUATION SUMMARY")
        print("="*70)
        print(f"  Mean Acceptance Length (MAL): {drafter_results['mean_mal']:.2f}")
        print(f"  Overall Acceptance Rate:     {drafter_results['overall_acceptance_rate']:.1f}%")
        print(f"  Target Pass Efficiency:       {drafter_results['target_pass_efficiency']:.2f}")
        print(f"  Mean TPS:                     {drafter_results['mean_tps']:.2f}")

    # Remove raw tokens from results (they're large)
    if "baseline" in results and "_raw_tokens" in results["baseline"]:
        del results["baseline"]["_raw_tokens"]
    if "_raw_tokens" in results["drafter"]:
        del results["drafter"]["_raw_tokens"]

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()