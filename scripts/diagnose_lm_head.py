#!/usr/bin/env python3
"""
P-EAGLE Diagnostic Script

Diagnoses why speculative decoding is failing (0% acceptance rate).
Checks:
1. lm_head is properly saved in feature files
2. lm_head is properly loaded from target model
3. Checkpoint contains correct lm_head
4. Drafter produces valid tokens (not gibberish)
"""

import argparse
import torch
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


def find_lm_head(model, model_name="model", depth=0):
    """Recursively find lm_head in a model, trying multiple possible paths."""
    results = []

    # Common paths for different model architectures
    paths_to_try = [
        (f"{model_name}.lm_head", lambda m: hasattr(m, 'lm_head') and m.lm_head is not None),
        (f"{model_name}.model.lm_head", lambda m: hasattr(m, 'model') and hasattr(m.model, 'lm_head') and m.model.lm_head is not None),
        (f"{model_name}.model.model.lm_head", lambda m: hasattr(m, 'model') and hasattr(m.model, 'model') and hasattr(m.model.model, 'lm_head') and m.model.model.lm_head is not None),
        (f"{model_name}.language_model.lm_head", lambda m: hasattr(m, 'language_model') and hasattr(m.language_model, 'lm_head') and m.language_model.lm_head is not None),
        # Gemma-3 multimodal structure
        (f"{model_name}.model.language_model.lm_head", lambda m: hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'lm_head') and m.model.language_model.lm_head is not None),
    ]

    for path_name, check_func in paths_to_try:
        if check_func(model):
            # Get the lm_head object
            if "model.model.model" in path_name:
                lm_obj = model.model.model.lm_head
            elif "model.model.language" in path_name:
                lm_obj = model.model.language_model.lm_head
            elif "model.model.lm_head" in path_name:
                lm_obj = model.model.lm_head
            elif "language_model.lm_head" in path_name:
                lm_obj = model.language_model.lm_head
            else:
                lm_obj = model.lm_head

            try:
                weight_shape = lm_obj.weight.shape
                results.append((path_name, weight_shape))
            except:
                pass

    return results


def diagnose_feature_files(feature_dir: str):
    """Check if feature files contain lm_head and if it's valid."""
    print("\n" + "="*70)
    print("DIAGNOSING FEATURE FILES")
    print("="*70)

    feature_path = Path(feature_dir)
    if not feature_path.exists():
        print(f"❌ Feature directory does not exist: {feature_dir}")
        return False

    shard_files = list(feature_path.glob("*_shard*.pt"))
    if not shard_files:
        print(f"❌ No feature files found in {feature_dir}")
        return False

    print(f"Found {len(shard_files)} feature shard files")

    all_have_lm_head = True
    for shard_file in shard_files[:5]:  # Check first 5 shards
        print(f"\nChecking: {shard_file.name}")
        try:
            data = torch.load(shard_file, map_location='cpu', weights_only=False)

            # Check lm_head
            has_lm_head = "lm_head" in data and data["lm_head"] is not None
            if has_lm_head:
                lm_head = data["lm_head"]
                if isinstance(lm_head, dict) and "weight" in lm_head:
                    weight_shape = lm_head["weight"].shape
                    print(f"  ✓ lm_head saved: shape {weight_shape}")

                    # Check if weights are non-zero (not random init)
                    weight = lm_head["weight"]
                    is_nonzero = (weight.abs() > 1e-6).any()
                    if is_nonzero:
                        print(f"  ✓ lm_head weights are non-zero (properly initialized)")
                    else:
                        print(f"  ⚠️  lm_head weights are ALL ZERO (might be random init)")
                else:
                    print(f"  ⚠️  lm_head is dict but missing 'weight' key")
            else:
                print(f"  ❌ lm_head NOT saved in feature file")
                print(f"  ❌ Training will use RANDOM lm_head -> WILL PRODUCE GIBBERISH")
                all_have_lm_head = False

            # Check vocab_size
            vocab_size = data.get("vocab_size", "NOT SAVED")
            hidden_size = data.get("hidden_size", "NOT SAVED")
            print(f"  vocab_size: {vocab_size}")
            print(f"  hidden_size: {hidden_size}")

        except Exception as e:
            print(f"  ❌ Error loading shard: {e}")
            all_have_lm_head = False

    return all_have_lm_head


def diagnose_target_model(target_model_name: str):
    """Check if target model has lm_head at expected locations."""
    print("\n" + "="*70)
    print("DIAGNOSING TARGET MODEL lm_head")
    print("="*70)

    print(f"\nLoading target model: {target_model_name}")

    try:
        target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print(f"  Model loaded successfully")
        print(f"  Model type: {type(target_model).__name__}")

        # Find lm_head at all possible locations
        lm_head_locations = find_lm_head(target_model, "target_model")

        if lm_head_locations:
            print(f"\n  Found lm_head at:")
            for path_name, weight_shape in lm_head_locations:
                print(f"    ✓ {path_name}: {weight_shape}")

            # Use the first (most specific) found
            best_path, best_shape = lm_head_locations[0]
            print(f"\n  Will use: {best_path}")
            print(f"  Vocab size: {best_shape[0]}")
            print(f"  Hidden size: {best_shape[1]}")

            return True
        else:
            print(f"\n  ❌ Could not find lm_head at any known location!")
            print(f"  ❌ Model type might not be supported: {type(target_model).__name__}")
            return False

    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return False


def diagnose_checkpoint(drafter_checkpoint: str, target_model_name: str):
    """Check if checkpoint has valid lm_head."""
    print("\n" + "="*70)
    print("DIAGNOSING DRAFTER CHECKPOINT")
    print("="*70)

    checkpoint_path = Path(drafter_checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory does not exist: {drafter_checkpoint}")
        return False

    # Check config.json
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        print(f"\nCheckpoint config:")
        print(f"  base_model: {config.get('base_model', 'N/A')}")
        print(f"  vocab_size: {config.get('vocab_size', 'N/A')}")
        print(f"  hidden_dim: {config.get('hidden_dim', 'N/A')}")
        print(f"  target_hidden_dim: {config.get('target_hidden_dim', 'N/A')}")

    # Check eagle_heads.pt
    heads_path = checkpoint_path / "eagle_heads.pt"
    if not heads_path.exists():
        print(f"❌ eagle_heads.pt not found")
        return False

    print(f"\nLoading eagle_heads.pt...")
    checkpoint = torch.load(heads_path, map_location='cpu')

    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Check target_lm_head
    has_target_lm_head = "target_lm_head" in checkpoint
    print(f"\ntarget_lm_head in checkpoint: {has_target_lm_head}")

    if has_target_lm_head:
        lm_head_state = checkpoint["target_lm_head"]
        if isinstance(lm_head_state, dict) and "weight" in lm_head_state:
            weight_shape = lm_head_state["weight"].shape
            print(f"  ✓ target_lm_head saved: shape {weight_shape}")

            # Check if weights are non-zero
            weight = lm_head_state["weight"]
            is_nonzero = (weight.abs() > 1e-6).any()
            std = weight.std().item()
            print(f"  Weight std: {std:.6f}")

            if is_nonzero and std > 0.01:  # Not all zeros and not near-zero
                print(f"  ✓ target_lm_head weights look properly initialized")
            else:
                print(f"  ⚠️  target_lm_head might be randomly initialized (std={std:.6f})")
                print(f"  ❌ This will cause 0% acceptance rate!")
        else:
            print(f"  ❌ target_lm_head has unexpected format")
    else:
        print(f"  ❌ target_lm_head NOT saved in checkpoint!")
        print(f"  ❌ Inference will use target model's lm_head instead")

    # Check if drafter has target_lm_head attribute
    try:
        from p_eagle.models.peagle_drafter import EagleDrafterModel
        drafter = EagleDrafterModel.load_checkpoint(drafter_checkpoint, device='cpu')

        has_attr = hasattr(drafter, 'target_lm_head') and drafter.target_lm_head is not None
        print(f"\nDrafter has target_lm_head attribute: {has_attr}")

        if has_attr:
            print(f"  Drafter target_lm_head shape: {drafter.target_lm_head.weight.shape}")

            # Check if target_lm_head is reasonable
            weight = drafter.target_lm_head.weight
            std = weight.std().item()
            max_val = weight.abs().max().item()
            print(f"  Weight std: {std:.6f}, max: {max_val:.4f}")

            if std > 0.01 and max_val > 0.1:
                print(f"  ✓ Drafter's target_lm_head looks valid")
            else:
                print(f"  ⚠️  Drafter's target_lm_head might be corrupted")
    except Exception as e:
        print(f"\nError loading drafter checkpoint: {e}")

    return has_target_lm_head


def diagnose_tokenizer(target_model_name: str, drafter_checkpoint: str):
    """Check if tokenizers are compatible."""
    print("\n" + "="*70)
    print("DIAGNOSING TOKENIZER COMPATIBILITY")
    print("="*70)

    # Load target tokenizer
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_vocab_size = len(target_tokenizer)
    print(f"\nTarget tokenizer vocab size: {target_vocab_size}")

    # Load from checkpoint config
    checkpoint_path = Path(drafter_checkpoint)
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        saved_vocab_size = config.get('vocab_size', None)
        if saved_vocab_size:
            print(f"Checkpoint saved vocab size: {saved_vocab_size}")
            if saved_vocab_size != target_vocab_size:
                print(f"  ❌ VOCAB SIZE MISMATCH!")
                print(f"  ❌ This will cause token ID confusion!")
                return False
            else:
                print(f"  ✓ Vocab sizes match")

    return True


def diagnose_mtp_heads(drafter_checkpoint: str):
    """Check if MTP heads are producing reasonable outputs."""
    print("\n" + "="*70)
    print("DIAGNOSING MTP HEADS")
    print("="*70)

    try:
        from p_eagle.models.peagle_drafter import EagleDrafterModel
        from transformers import AutoTokenizer

        drafter = EagleDrafterModel.load_checkpoint(drafter_checkpoint, device='cpu')
        tokenizer = AutoTokenizer.from_pretrained(drafter_checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test with a simple prompt
        prompt = "Hello"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        print(f"\nTesting with prompt: '{prompt}'")
        print(f"Input tokens: {input_ids.tolist()}")

        # Get base embeddings
        inputs_embeds = drafter.base_model.get_input_embeddings()(input_ids)
        print(f"Input embeddings shape: {inputs_embeds.shape}")

        # Create dummy target hidden (zeros for testing)
        batch_size, seq_len = input_ids.shape
        target_hidden = torch.zeros(batch_size, seq_len, drafter.target_hidden_dim)

        # Forward pass
        with torch.no_grad():
            outputs = drafter(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                target_hidden=target_hidden,
                is_training=False
            )

        mtp_predictions = outputs["mtp_predictions"]
        print(f"\nMTP head predictions:")
        for i, pred in enumerate(mtp_predictions):
            print(f"  Head {i+1}: shape {pred.shape}")

            # Project to vocab and check distribution
            if hasattr(drafter, 'target_lm_head') and drafter.target_lm_head is not None:
                lm_head = drafter.target_lm_head
            else:
                print(f"    ⚠️  No target_lm_head available to test")
                continue

            logits = lm_head(pred)
            probs = torch.softmax(logits, dim=-1)

            top_prob, top_token = probs.max(dim=-1)
            top_token_str = tokenizer.decode([top_token.item()])

            print(f"    Top token: {top_token.item()} ('{top_token_str}') with prob {top_prob.item():.4f}")

            # Check entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            print(f"    Entropy: {entropy.item():.4f}")

            if top_prob.item() < 0.1:
                print(f"    ⚠️  Low confidence - drafter might not be learning well")

        return True

    except Exception as e:
        print(f"  ❌ Error testing MTP heads: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Diagnostic Script")
    parser.add_argument("--feature_dir", required=True, help="Path to feature directory")
    parser.add_argument("--target_model", required=True, help="Target model name (e.g., google/gemma-3-4b-it)")
    parser.add_argument("--drafter_checkpoint", required=True, help="Path to drafter checkpoint")

    args = parser.parse_args()

    print("="*70)
    print("P-EAGLE SPECULATIVE DECODING DIAGNOSTIC")
    print("="*70)

    results = {}

    # 1. Check feature files
    results['feature_files'] = diagnose_feature_files(args.feature_dir)

    # 2. Check target model
    results['target_model'] = diagnose_target_model(args.target_model)

    # 3. Check checkpoint
    results['checkpoint'] = diagnose_checkpoint(args.drafter_checkpoint, args.target_model)

    # 4. Check tokenizer compatibility
    results['tokenizer'] = diagnose_tokenizer(args.target_model, args.drafter_checkpoint)

    # 5. Check MTP heads
    results['mtp_heads'] = diagnose_mtp_heads(args.drafter_checkpoint)

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Speculative decoding should work")
    else:
        print("❌ SOME CHECKS FAILED - This explains 0% acceptance rate")
        print("\nPossible causes:")
        if not results.get('feature_files'):
            print("  - Feature files don't contain lm_head")
            print("  - FIX: Re-extract features with the updated feature_extractor.py")
        if not results.get('checkpoint'):
            print("  - Checkpoint doesn't contain valid target_lm_head")
            print("  - FIX: Re-train with the updated trainer.py")
        if not results.get('tokenizer'):
            print("  - Vocabulary size mismatch between training and inference")
    print("="*70)


if __name__ == "__main__":
    main()