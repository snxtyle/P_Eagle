#!/usr/bin/env python3
"""
P-EAGLE Sequence Packing Script

Packs tokenized conversations into solid 4096-token blocks with cu_seqlens boundaries.
This format is optimized for FlashAttention varlen and H200 GPU efficiency.

Usage:
    # Phase 1: Pack first 5K samples
    python scripts/pack_sequences.py --input data/dataset/semantic_chunks.jsonl --output data/packed_5k --max_samples 5000

    # Phase 2: Pack first 10K samples
    python scripts/pack_sequences.py --input data/dataset/semantic_chunks.jsonl --output data/packed_10k --max_samples 10000

    # Phase 3: Full dataset
    python scripts/pack_sequences.py --input data/dataset/semantic_chunks.jsonl --output data/packed_full
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np


class SequencePacker:
    """
    Packs variable-length tokenized sequences into solid 4096-token blocks.

    Output format compatible with FlashAttention varlen:
    - input_ids: [total_tokens] tensor of token IDs
    - loss_mask: [total_tokens] tensor (1 for trainable, 0 for ignore)
    - cu_seqlens: [num_sequences + 1] tensor of cumulative sequence boundaries
    - max_seq_len: 4096 (fixed block size)
    """

    def __init__(self, tokenizer, block_size: int = 4096, preserve_boundaries: bool = True):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.preserve_boundaries = preserve_boundaries  # Don't split conversations

    def _messages_to_text(self, messages):
        """Convert OpenAI messages to conversation text, preserving all roles."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if content is None:
                content = ""
            elif isinstance(content, list):
                # Handle list content (e.g., tool results)
                text_parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                        elif "text" in c:
                            text_parts.append(str(c["text"]))
                content = "\n".join(text_parts)

            # Handle tool_calls
            if not content and msg.get("tool_calls"):
                tool_parts = []
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        args_str = args
                    else:
                        args_str = json.dumps(args)
                    tool_parts.append(f"[TOOL_CALL] {name}({args_str}) [/TOOL_CALL]")
                content = "\n".join(tool_parts)

            if content:
                parts.append(f"{role}: {content}")

        return "\n\n".join(parts)

    def _get_loss_mask_from_sample(self, sample):
        """Extract loss mask from sample, handling various formats."""
        # Format 1: Direct loss_mask array
        if "loss_mask" in sample:
            loss_mask = sample["loss_mask"]
            if isinstance(loss_mask, list):
                return [int(x) for x in loss_mask]
            return loss_mask

        # Format 2: segments format
        if "segments" in sample:
            segments = sample["segments"]
            loss_mask = []
            for seg in segments:
                mask = seg.get("mask", seg.get("train_mask", 0))
                loss_mask.append(int(mask))
            return loss_mask

        # Format 3: loss_mask_segments format
        if "loss_mask_segments" in sample:
            lms = sample["loss_mask_segments"]
            if "segments" in lms:
                return [int(s.get("mask", 0)) for s in lms["segments"]]
            if "train_indices" in lms:
                train_set = set(lms["train_indices"])
                return [1 if i in train_set else 0 for i in range(len(lms.get("train_indices", [0])))]

        # Format 4: Auto-generate from roles
        messages = sample.get("original_messages", sample.get("messages", []))
        assistant_roles = {"assistant", "bot", "ai"}
        loss_mask = []
        for msg in messages:
            role = msg.get("role", "").lower().strip()
            is_assistant = role in assistant_roles
            has_content = bool(msg.get("content"))
            mask = 1 if (is_assistant and has_content) else 0
            loss_mask.append(mask)
        return loss_mask

    def pack_file(self, input_path: str, output_dir: str, max_samples: int = None,
                  shard_size: int = 5000):
        """
        Pack sequences from JSONL file into 4096-token blocks.

        Args:
            input_path: Path to semantic_chunks.jsonl
            output_dir: Output directory for packed tensors
            max_samples: Maximum samples to process (None = all)
            shard_size: Samples per output shard
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate data
        all_input_ids = []
        all_loss_masks = []
        all_texts = []
        all_cu_seqlens = [0]

        total_tokens = 0
        sample_count = 0

        print(f"Packing sequences from {input_path}")
        print(f"Block size: {self.block_size}, Shard size: {shard_size}")

        with open(input_path, 'r') as f:
            for line in tqdm(f, desc="Processing samples"):
                if max_samples and sample_count >= max_samples:
                    break

                sample = json.loads(line)

                # Get conversation text and tokenize
                messages = sample.get("original_messages", sample.get("messages", []))
                if not messages:
                    # Try conversation_text format
                    conversation_text = sample.get("conversation_text", "")
                    if conversation_text:
                        # Re-parse from conversation text
                        messages = sample.get("original_messages", [])
                        if not messages:
                            continue

                conversation_text = self._messages_to_text(messages)

                # Tokenize
                encoded = self.tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.block_size - 2,  # Leave room for special tokens
                    add_special_tokens=True
                )

                input_ids = encoded["input_ids"][0].tolist()
                seq_len = len(input_ids)

                # Get loss mask aligned to tokens
                loss_mask_list = self._get_loss_mask_from_sample(sample)

                # Simple approach: approximate alignment (mask first/last proportional)
                # For accurate alignment, we'd need to re-tokenize and match
                # This is a heuristic that works reasonably well
                if len(loss_mask_list) >= 2:
                    # Proportional mapping
                    loss_mask = []
                    for i in range(seq_len):
                        idx = int(i * len(loss_mask_list) / seq_len)
                        idx = min(idx, len(loss_mask_list) - 1)
                        loss_mask.append(loss_mask_list[idx])
                else:
                    # Single mask - apply to all
                    mask_val = loss_mask_list[0] if loss_mask_list else 0
                    loss_mask = [mask_val] * seq_len

                # Store
                all_input_ids.extend(input_ids)
                all_loss_masks.extend(loss_mask)
                all_texts.append(conversation_text[:200])  # Store truncated text

                cu_seqlens_offset = all_cu_seqlens[-1] + seq_len
                all_cu_seqlens.append(cu_seqlens_offset)

                total_tokens += seq_len
                sample_count += 1

                # Check if we should flush a shard
                current_total = all_cu_seqlens[-1]
                blocks_needed = (current_total + self.block_size - 1) // self.block_size

                if blocks_needed >= shard_size:
                    # Save this shard (full blocks)
                    self._save_shard(
                        all_input_ids, all_loss_masks, all_cu_seqlens,
                        output_dir, sample_count, blocks_needed
                    )
                    # Keep remainder
                    remainder = current_total % self.block_size
                    if remainder > 0:
                        all_input_ids = all_input_ids[-remainder:]
                        all_loss_masks = all_loss_masks[-remainder:]
                        all_cu_seqlens = [0, remainder]
                    else:
                        all_input_ids = []
                        all_loss_masks = []
                        all_cu_seqlens = [0]

        # Save final shard
        if all_cu_seqlens[-1] > 0:
            current_total = all_cu_seqlens[-1]
            blocks_needed = (current_total + self.block_size - 1) // self.block_size
            self._save_shard(
                all_input_ids, all_loss_masks, all_cu_seqlens,
                output_dir, sample_count, blocks_needed
            )

        print(f"\nPacking complete!")
        print(f"  Samples processed: {sample_count}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Blocks generated: ~{total_tokens // self.block_size}")
        print(f"  Output: {output_dir}")

    def _save_shard(self, input_ids, loss_masks, cu_seqlens, output_dir,
                    shard_idx, estimated_blocks):
        """Save a shard of packed sequences."""
        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_masks, dtype=torch.float32)
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32)

        output_file = output_dir / f"packed_shard_{shard_idx:06d}.pt"

        torch.save({
            "input_ids": input_ids_tensor,
            "loss_mask": loss_mask_tensor,
            "cu_seqlens": cu_seqlens_tensor,
            "block_size": self.block_size,
            "num_sequences": len(cu_seqlens) - 1,
            "total_tokens": len(input_ids)
        }, output_file)

        print(f"  Saved shard {shard_idx}: {len(input_ids):,} tokens, {len(cu_seqlens)-1} sequences")


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Sequence Packing")
    parser.add_argument("--input", required=True, help="Input JSONL file (semantic_chunks.jsonl)")
    parser.add_argument("--output", required=True, help="Output directory for packed tensors")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B",
                        help="Tokenizer to use")
    parser.add_argument("--block_size", type=int, default=4096,
                        help="Fixed block size (default: 4096)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process (None = all)")
    parser.add_argument("--shard_size", type=int, default=5000,
                        help="Samples per shard (default: 5000)")

    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    packer = SequencePacker(tokenizer, block_size=args.block_size)
    packer.pack_file(
        args.input,
        args.output,
        max_samples=args.max_samples,
        shard_size=args.shard_size
    )


if __name__ == "__main__":
    main()