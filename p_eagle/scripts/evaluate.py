#!/usr/bin/env python3
"""
P-EAGLE Evaluation Script - Proper Speculative Decoding

Key metrics:
1. Speed: TPS comparison (baseline vs drafter)
2. Acceptance: Mean Acceptance Length (MAL), acceptance rate per head
3. Quality: Perplexity, token distribution similarity
"""

import argparse
import json
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

torch._inductor.config.triton.cudagraphs = False
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"


def evaluate_baseline(target_model_name: str, prompts: List[str], max_tokens: int = 100,
                      temperature: float = 0.7, top_p: float = 0.9) -> Dict:
    """Evaluate raw target model baseline."""
    print("\n" + "="*70)
    print("  BASELINE: Raw Target Model")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {target_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    results = []
    total_time = 0
    total_tokens = 0
    all_perplexities = []

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

        generated_ids = output[0, original_length:]
        tokens_generated = len(generated_ids)

        # Calculate perplexity of generated tokens
        with torch.no_grad():
            full_output = model(output)
            logits = full_output.logits[0, original_length-1:-1]
            probs = F.softmax(logits, dim=-1)
            perplexity = float(torch.exp(F.cross_entropy(logits, generated_ids)))

        tps = tokens_generated / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
            "tokens": tokens_generated,
            "time": elapsed,
            "tps": tps,
            "perplexity": perplexity
        })

        total_time += elapsed
        total_tokens += tokens_generated
        all_perplexities.append(perplexity)
        print(f"{tokens_generated} tokens, {elapsed:.2f}s, {tps:.1f} tps, ppl={perplexity:.2f}")

    return {
        "model": target_model_name,
        "total_samples": len(prompts),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "mean_tps": total_tokens / total_time if total_time > 0 else 0,
        "mean_perplexity": np.mean(all_perplexities),
        "samples": results
    }


def evaluate_speculative(drafter_checkpoint: str, target_model_name: str,
                         prompts: List[str], max_tokens: int = 100,
                         temperature: float = 0.7, top_p: float = 0.9) -> Dict:
    """Evaluate with TRUE speculative decoding (no teacher forcing).

    In true EAGLE speculative decoding:
    1. Drafter generates K draft tokens based on its OWN predictions (NOT target's hidden)
    2. Target verifies ALL K drafts in parallel
    3. Accepted tokens are kept; first rejection triggers single target token gen
    """
    from p_eagle.models.peagle_drafter import EagleDrafterModel

    print("\n" + "="*70)
    print("  P-EAGLE: Speculative Decoding")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load target model for verification
    print(f"Loading target model: {target_model_name}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).to("cuda")
    target_model.eval()

    # Load drafter
    print(f"Loading drafter from {drafter_checkpoint}...")
    drafter = EagleDrafterModel.load_checkpoint(drafter_checkpoint, device="cuda")
    drafter.eval()

    # Get the target model's lm_head for token generation
    # The drafter outputs hidden states that match the target's hidden_dim
    # We use the target model's lm_head to convert hidden states to logits
    if hasattr(target_model, 'lm_head'):
        target_lm_head = target_model.lm_head
    elif hasattr(target_model, 'model') and hasattr(target_model.model, 'lm_head'):
        target_lm_head = target_model.model.lm_head
    else:
        raise ValueError("Could not find lm_head in target model")

    # Create a projection from drafter's output dim to target's hidden dim
    # The drafter (gemma-3-270m-it) outputs 640-dim hidden states
    # We need to project to the target's 2560-dim for the lm_head
    drafter_output_dim = 640  # gemma-3-270m-it's embedding/output dimension
    target_hidden_dim = 2560  # What the target model expects (from training)

    print(f"Drafter output dim: {drafter_output_dim}")
    print(f"Target hidden dim: {target_hidden_dim}")

    # Load projection weights from drafter if available
    if hasattr(drafter, 'target_hidden_proj'):
        with torch.no_grad():
            src_weight = drafter.target_hidden_proj.weight  # [640, 2560] (output, input)
            src_bias = drafter.target_hidden_proj.bias  # [640]
            print(f"Source projection: {src_weight.shape}, bias: {src_bias.shape if src_bias is not None else 'None'}")

            # Create projection layer: 640 -> 2560
            # But we need 2560 -> 640 for lm_head (transpose of target_hidden_proj)
            # Actually, the mtp_predictions are 2560-dim (target's hidden dim)
            # We need to project from 2560 -> 640 for the lm_head
            lm_head_projection = torch.nn.Linear(target_hidden_dim, drafter_output_dim, dtype=torch.bfloat16).to("cuda")
            with torch.no_grad():
                # Transpose: src is [640, 2560], dest needs [640, 2560]
                lm_head_projection.weight.copy_(src_weight)  # Already correct shape!
                if src_bias is not None:
                    lm_head_projection.bias.copy_(src_bias)
        print(f"Loaded projection weights: {target_hidden_dim} -> {drafter_output_dim}")
    else:
        print(f"Warning: No projection weights found, using random init")
        lm_head_projection = torch.nn.Linear(target_hidden_dim, drafter_output_dim, dtype=torch.bfloat16).to("cuda")

    speculation_depth = drafter.speculation_depth
    print(f"Speculation depth (K): {speculation_depth}")

    results = []
    all_acceptance_by_head = {h: [] for h in range(1, speculation_depth + 2)}  # +2 for >K and 0
    all_mal = []
    total_drafted = 0
    total_accepted = 0
    all_perplexities = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] Speculating...", end=" ", flush=True)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        original_length = input_ids.shape[1]
        generated = input_ids.clone()

        start = time.time()
        acceptance_counts = []

        # Speculative decoding loop
        for _ in range(max_tokens):
            if generated.shape[1] >= original_length + max_tokens:
                break

            # Step 1: Drafter generates K draft tokens (using its OWN predictions)
            with torch.no_grad():
                drafter_outputs = drafter.forward(
                    input_ids=generated,
                    target_hidden=None,  # IMPORTANT: No teacher forcing!
                    is_training=False
                )
                mtp_predictions = drafter_outputs["mtp_predictions"]

            # Convert hidden states to K draft tokens
            # MTP heads output hidden states matching target's dimension (2560)
            # Use target's lm_head directly - NO projection needed!
            draft_tokens = []
            for k in range(min(speculation_depth, max_tokens - (generated.shape[1] - original_length))):
                pred_hidden = mtp_predictions[k]  # [1, seq, 2560]
                # Use target's lm_head directly - pred_hidden is already 2560-dim
                logits = target_lm_head(pred_hidden[:, -1:, :])  # [1, 1, vocab]
                token_id = torch.argmax(logits, dim=-1).item()
                draft_tokens.append(token_id)

            if not draft_tokens:
                break

            # Step 2: Target verifies ALL drafts in parallel
            draft_tensor = torch.tensor([draft_tokens], device="cuda")
            verify_input = torch.cat([generated, draft_tensor], dim=1)

            with torch.no_grad():
                verify_outputs = target_model(verify_input)
                verify_logits = verify_outputs.logits[0, generated.shape[1]-1:, :]  # [K, vocab]

            # Step 3: Accept/reject tokens
            accepted_count = 0
            for j, draft_token in enumerate(draft_tokens):
                target_token = torch.argmax(verify_logits[j]).item()
                if draft_token == target_token:
                    accepted_count += 1
                    total_accepted += 1
                else:
                    # Rejection - stop accepting further tokens
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

        elapsed = time.time() - start
        tokens_generated = generated.shape[1] - original_length
        mal = np.mean(acceptance_counts) if acceptance_counts else 0
        all_mal.append(mal)

        # Calculate perplexity of generated tokens
        with torch.no_grad():
            full_output = target_model(generated)
            logits = full_output.logits[0, original_length-1:-1]
            generated_ids = generated[0, original_length:]
            if len(generated_ids) > 0 and logits.shape[0] > 0:
                perplexity = float(torch.exp(F.cross_entropy(logits, generated_ids[:logits.shape[0]])))
            else:
                perplexity = 0
        all_perplexities.append(perplexity)

        tps = tokens_generated / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
            "tokens": tokens_generated,
            "mal": mal,
            "time": elapsed,
            "tps": tps,
            "perplexity": perplexity,
            "accepted": sum(acceptance_counts),
            "drafted": total_drafted
        })

        print(f"{tokens_generated} tokens, MAL={mal:.2f}, {tps:.1f} tps, ppl={perplexity:.2f}")

    # Calculate acceptance rate per head
    acceptance_by_head_pct = {
        h: len([x for x in all_acceptance_by_head[h]]) / len(prompts) * 100
        if len(all_acceptance_by_head[h]) > 0 else 0
        for h in range(1, speculation_depth + 2)
    }

    return {
        "drafter_checkpoint": drafter_checkpoint,
        "target_model": target_model_name,
        "speculation_depth": speculation_depth,
        "total_samples": len(prompts),
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "overall_acceptance_rate": total_accepted / total_drafted * 100 if total_drafted > 0 else 0,
        "mean_mal": np.mean(all_mal),
        "mean_perplexity": np.mean(all_perplexities),
        "acceptance_by_head": acceptance_by_head_pct,
        "samples": results
    }


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Speculative Decoding Evaluation")
    parser.add_argument("--drafter_checkpoint", default="./checkpoints_peagle/best_model",
                       help="Path to trained drafter checkpoint")
    parser.add_argument("--target_model", default="google/gemma-2b-it",
                       help="Target model (MUST match the one used for feature extraction)")
    parser.add_argument("--test_prompts", default=None,
                       help="JSON file with test prompts")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Max tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--baseline", action="store_true", default=True,
                       help="Also evaluate baseline (no drafter)")
    parser.add_argument("--output", default="eval_results.json",
                       help="Output file for results")

    args = parser.parse_args()

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
    if args.baseline:
        baseline_results = evaluate_baseline(args.target_model, prompts, args.max_tokens,
                                            args.temperature, args.top_p)
        results["baseline"] = baseline_results
        baseline_tps = baseline_results["mean_tps"]
        baseline_ppl = baseline_results["mean_perplexity"]

    # Evaluate with drafter
    drafter_results = evaluate_speculative(args.drafter_checkpoint, args.target_model,
                                          prompts, args.max_tokens, args.temperature, args.top_p)
    results["drafter"] = drafter_results
    drafter_tps = drafter_results["mean_tps"]
    drafter_ppl = drafter_results["mean_perplexity"]

    # Calculate speedup
    if args.baseline:
        speedup = drafter_tps / baseline_tps if baseline_tps > 0 else 1.0
        ppl_delta = drafter_ppl - baseline_ppl
        ppl_ratio = drafter_ppl / baseline_ppl if baseline_ppl > 0 else 1.0

        print("\n" + "="*70)
        print("  EVALUATION SUMMARY")
        print("="*70)
        print(f"\n{'Metric':<30} {'Baseline':<15} {'P-EAGLE':<15} {'Speedup':<10}")
        print("-" * 70)
        print(f"{'Tokens/Second (TPS)':<30} {baseline_tps:<15.1f} {drafter_tps:<15.1f} {speedup:<10.2f}x")
        print(f"{'Perplexity':<30} {baseline_ppl:<15.2f} {drafter_ppl:<15.2f} {ppl_ratio:<10.2f}x")

        print(f"\n{'Acceptance Metrics':<40}")
        print("-" * 70)
        print(f"  Mean Acceptance Length (MAL):    {drafter_results['mean_mal']:.2f}")
        print(f"  Overall Acceptance Rate:         {drafter_results['overall_acceptance_rate']:.1f}%")
        print(f"\n  Acceptance by Head Position:")
        for head, rate in sorted(drafter_results['acceptance_by_head'].items()):
            print(f"    Head {head}: {rate:.1f}%")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"  Results saved to: {args.output}")
    print("="*70)

    if args.baseline:
        if speedup > 1.0:
            print(f"\n✓ SPEEDUP: {speedup:.2f}x faster with drafter")
        else:
            print(f"\n✗ No speedup - check if drafter acceptance rate is sufficient")


if __name__ == "__main__":
    main()