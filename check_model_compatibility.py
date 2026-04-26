#!/usr/bin/env python3
"""
Check model compatibility for P-EAGLE training.
Ensures drafter and target have compatible vocabularies.
"""

import argparse
import sys
from transformers import AutoTokenizer, AutoConfig


def check_vocab_compatibility(target_model: str, drafter_model: str) -> bool:
    """Check if two models have compatible vocabularies."""
    print(f"\nChecking vocabulary compatibility...")
    print(f"  Target:  {target_model}")
    print(f"  Drafter: {drafter_model}")

    try:
        tok_target = AutoTokenizer.from_pretrained(target_model, token=True)
        tok_drafter = AutoTokenizer.from_pretrained(drafter_model, token=True)
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return False

    vocab_target = len(tok_target)
    vocab_drafter = len(tok_drafter)

    print(f"\n  Target vocab size:  {vocab_target}")
    print(f"  Drafter vocab size: {vocab_drafter}")

    # Test encoding
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "Payment processing",
        "<agent>system</agent>",
        "function_call()"
    ]

    mismatches = 0
    for text in test_texts:
        enc_target = tok_target.encode(text, add_special_tokens=False)
        enc_drafter = tok_drafter.encode(text, add_special_tokens=False)
        if enc_target != enc_drafter:
            mismatches += 1
            print(f"\n  MISMATCH for: '{text[:50]}'")
            print(f"    Target:  {enc_target[:10]}...")
            print(f"    Drafter: {enc_drafter[:10]}...")

    if mismatches == 0:
        print(f"\n  Tokenization: IDENTICAL (all {len(test_texts)} test strings match)")
        return True
    else:
        print(f"\n  Tokenization: MISMATCH ({mismatches}/{len(test_texts)} test strings differ)")
        return False


def get_hidden_dims(model_name: str) -> int:
    """Get hidden dimension of a model."""
    config = AutoConfig.from_pretrained(model_name, token=True)
    return config.hidden_size


def recommend_model_pair(domain: str = "general"):
    """Recommend compatible model pairs."""
    print(f"\n{'='*60}")
    print(f"RECOMMENDED MODEL PAIRS FOR {domain.upper()}")
    print(f"{'='*60}")

    pairs = {
        "general": [
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Same vocab, proven speedup"),
            ("google/gemma-2b-it", "unsloth/gemma-2-2b", "Same vocab (Gemma series)"),
            ("meta-llama/Llama-2-7b-chat-hf", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Different vocab - needs vocab mapping"),
        ],
        "code": [
            ("codellama/CodeLlama-7b-Python-hf", "codellama/CodeLlama-7b-Python-hf", "Use same model with LoRA"),
            ("Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen2.5-Coder-1.5B", "Same vocab"),
        ],
        "multilingual": [
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Excellent multilingual support"),
        ]
    }

    for target, drafter, notes in pairs.get(domain, pairs["general"]):
        print(f"\n  Target:  {target}")
        print(f"  Drafter: {drafter}")
        print(f"  Notes:   {notes}")


def main():
    parser = argparse.ArgumentParser(description="Check P-EAGLE model compatibility")
    parser.add_argument("--target", help="Target model name")
    parser.add_argument("--drafter", help="Drafter model name")
    parser.add_argument("--recommend", action="store_true", help="Show recommended pairs")
    parser.add_argument("--domain", default="general", choices=["general", "code", "multilingual"])

    args = parser.parse_args()

    if args.recommend:
        recommend_model_pair(args.domain)
        return 0

    if not args.target or not args.drafter:
        print("Error: --target and --drafter required (or use --recommend)")
        return 1

    print(f"{'='*60}")
    print(f"P-EAGLE Model Compatibility Check")
    print(f"{'='*60}")

    # Check vocab
    vocab_ok = check_vocab_compatibility(args.target, args.drafter)

    # Check hidden dims
    print(f"\nChecking hidden dimensions...")
    try:
        target_dim = get_hidden_dims(args.target)
        drafter_dim = get_hidden_dims(args.drafter)
        print(f"  Target hidden dim:  {target_dim}")
        print(f"  Drafter hidden dim: {drafter_dim}")
        print(f"  Projection needed:  {target_dim != drafter_dim}")
    except Exception as e:
        print(f"  Error: {e}")

    # Summary
    print(f"\n{'='*60}")
    if vocab_ok:
        print("COMPATIBLE - These models should work well together")
        return 0
    else:
        print("WARNING - Vocabulary mismatch detected!")
        print("\nRecommendations:")
        print("  1. Use models from the same family (e.g., both Gemma or both Qwen)")
        print("  2. Use same model with LoRA for drafter")
        print("  3. Implement vocabulary mapping (advanced)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
