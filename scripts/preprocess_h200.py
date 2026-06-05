#!/usr/bin/env python3
"""
H200-Optimized Sequence Packing Pipeline

Professional-grade preprocessing for P-EAGLE speculative decoding training:
1. Load Claude Code conversations from JSONL
2. Format using chat template (Qwen-compatible <|im_start|>/<|im_end|>)
3. Apply Sliding Window with System Prompt Anchoring
4. Pack into 4096-token blocks with cu_seqlens
5. Save as binary .pt tensors for fast streaming

Usage:
    python scripts/preprocess_h200.py \
        --input data/openai_format.jsonl \
        --output data/packed_h200 \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --max_seq_len 4096 \
        --shard_size 5000
"""

import json
import os
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ConversationChunk:
    """A chunk of conversation with token IDs."""
    token_ids: List[int]
    conv_id: str
    chunk_idx: int
    has_system: bool

    @property
    def length(self) -> int:
        return len(self.token_ids)


class ClaudeToQwenFormatter:
    """
    Convert Claude Code messages to Qwen chat format for tokenization.

    Claude format: {"role": "system/user/assistant/tool", "content": "..."}
    Qwen format:   <|im_start|>role\ncontent<|im_end|>
    """

    ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }

    def format(self, messages: List[Dict]) -> str:
        """
        Format messages using Qwen chat template.

        Example output:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if content is None:
                continue

            # Map role if needed
            role = self.ROLE_MAP.get(role, role)

            # Format with Qwen template
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(parts)

    def extract_system_prompt(self, messages: List[Dict]) -> Optional[List[int]]:
        """Extract system prompt tokens for anchoring."""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    return f"<|im_start|>system\n{content}<|im_end|>"
        return None


class SlidingWindowChunker:
    """
    Apply sliding window chunking with system prompt anchoring.

    Algorithm:
    1. Tokenize full conversation
    2. If short (<=max_seq_len): return as single chunk
    3. If long: create overlapping windows with system prompt prepended to each
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 4096,
        overlap_ratio: float = 0.5,
        stride: int = None
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.overlap_ratio = overlap_ratio
        # Default stride: 50% of max_seq_len
        self.stride = stride or int(max_seq_len * (1 - overlap_ratio))
        self.pad_token_id = tokenizer.pad_token_id or 0

        # Get special token IDs
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Chat template markers
        try:
            self.im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
            self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        except:
            self.im_start = None
            self.im_end = None

    def _format_conversation(self, messages: List[Dict], formatter: ClaudeToQwenFormatter) -> str:
        """Format conversation using Qwen template."""
        return formatter.format(messages)

    def chunk_conversation(
        self,
        messages: List[Dict],
        conv_id: str,
        formatter: ClaudeToQwenFormatter
    ) -> List[ConversationChunk]:
        """
        Apply sliding window with system prompt anchoring.

        For long conversations:
        - Chunk 0: [System Prompt + Start] [Turns 1..N] -> max_seq_len tokens
        - Chunk 1: [System Prompt] [End of Chunk 0] [Turns N+1..] -> max_seq_len tokens
        - Chunk N: [System Prompt] [End of Chunk N-1] [Turns ...] -> max_seq_len tokens

        System prompt is ALWAYS prepended to maintain tool definitions.
        """
        chunks = []

        # Format full conversation
        full_text = self._format_conversation(messages, formatter)

        # Get system prompt text
        system_text = formatter.extract_system_prompt(messages)

        # Tokenize with special tokens
        kwargs = {"add_special_tokens": True}
        if self.bos_token_id:
            full_ids = [self.bos_token_id]
            full_ids += self.tokenizer.encode(full_text, add_special_tokens=False)
            if self.eos_token_id:
                full_ids.append(self.eos_token_id)
        else:
            full_ids = self.tokenizer.encode(full_text, **kwargs)

        total_len = len(full_ids)

        # If fits in window, return as single chunk
        if total_len <= self.max_seq_len:
            return [ConversationChunk(
                token_ids=full_ids,
                conv_id=conv_id,
                chunk_idx=0,
                has_system=system_text is not None
            )]

        # Tokenize system prompt if exists (for prepending to later chunks)
        system_ids = []
        if system_text:
            if self.bos_token_id:
                system_ids = [self.bos_token_id]
                system_ids += self.tokenizer.encode(system_text, add_special_tokens=False)
            else:
                system_ids = self.tokenizer.encode(system_text, add_special_tokens=True)
            system_len = len(system_ids)
        else:
            system_len = 0

        # Sliding window
        start = 0
        chunk_idx = 0

        while start < total_len:
            # Calculate end position (leave room for system prompt if not first chunk)
            if chunk_idx == 0:
                # First chunk: use full window
                end = min(start + self.max_seq_len, total_len)
                chunk_ids = full_ids[start:end]
                has_system = system_text is not None
            else:
                # Subsequent chunks: prepend system prompt
                available_for_content = self.max_seq_len - system_len
                if available_for_content <= 0:
                    # System prompt alone exceeds window - skip
                    break

                end = min(start + available_for_content, total_len)
                chunk_ids = system_ids + full_ids[start:end]
                has_system = True

            # Ensure chunk doesn't exceed max_seq_len
            chunk_ids = chunk_ids[:self.max_seq_len]

            if len(chunk_ids) >= 256:  # Minimum chunk length
                chunks.append(ConversationChunk(
                    token_ids=chunk_ids,
                    conv_id=conv_id,
                    chunk_idx=chunk_idx,
                    has_system=has_system
                ))

            chunk_idx += 1
            start += self.stride

            # Safety: prevent infinite loop
            if self.stride == 0:
                break

        return chunks


class BlockPacker:
    """
    Pack variable-length chunks into fixed-size blocks for 100% GPU utilization.

    Uses First-Fit Decreasing (FFD) bin packing:
    1. Sort chunks by length (descending)
    2. Pack into blocks, filling each to capacity
    3. Generate cu_seqlens metadata for FlashAttention varlen
    """

    def __init__(self, block_size: int = 4096):
        self.block_size = block_size

    def pack(
        self,
        chunks: List[ConversationChunk]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Pack chunks into fixed-size blocks.

        Returns:
            input_ids: [num_blocks, block_size] tensor of token IDs
            cu_seqlens: List of [num_sequences + 1] cumulative sequence lengths
            loss_masks: [num_blocks, block_size] tensor (1 for trainable tokens)
        """
        # Sort by length (largest first) for better packing
        sorted_chunks = sorted(chunks, key=lambda x: x.length, reverse=True)

        # Group chunks into blocks
        blocks = []  # List of list of chunks
        block_occupancy = []  # Current token count in each block

        for chunk in sorted_chunks:
            chunk_len = chunk.length

            # Find first block that fits this chunk
            placed = False
            for i, occ in enumerate(block_occupancy):
                if occ + chunk_len <= self.block_size:
                    blocks[i].append(chunk)
                    block_occupancy[i] += chunk_len
                    placed = True
                    break

            # If no fitting block, create new one
            if not placed:
                blocks.append([chunk])
                block_occupancy.append(chunk_len)

        # Create tensors
        all_input_ids = []
        all_cu_seqlens = []
        all_loss_masks = []

        for block_chunks in blocks:
            # Flatten block
            block_tokens = []
            cu_seqlens = [0]
            loss_mask = []

            for chunk in block_chunks:
                block_tokens.extend(chunk.token_ids)
                # Loss on assistant and tool responses (after user)
                # For simplicity: mask assistant/tool content
                loss_mask.extend([1] * chunk.length)
                cu_seqlens.append(len(block_tokens))

            # Pad to block_size
            while len(block_tokens) < self.block_size:
                block_tokens.append(0)  # pad token
                loss_mask.append(0)

            all_input_ids.append(torch.tensor(block_tokens, dtype=torch.long))
            all_cu_seqlens.append(torch.tensor(cu_seqlens, dtype=torch.int32))
            all_loss_masks.append(torch.tensor(loss_mask, dtype=torch.float32))

        return all_input_ids, all_cu_seqlens, all_loss_masks


def load_conversations(input_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load conversations from JSONL file."""
    conversations = []

    with open(input_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Loading conversations", unit=" lines")):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line.strip())
                messages = data.get("messages", [])
                if messages and len(messages) >= 2:
                    conversations.append({
                        "id": data.get("id", f"conv_{i}"),
                        "messages": messages
                    })
            except json.JSONDecodeError:
                continue

    return conversations


def save_shard(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    loss_mask: torch.Tensor,
    output_dir: Path,
    shard_idx: int
):
    """Save a shard as binary .pt tensors."""
    output_file = output_dir / f"packed_shard_{shard_idx:06d}.pt"

    torch.save({
        "input_ids": input_ids,
        "cu_seqlens": cu_seqlens,
        "loss_mask": loss_mask,
        "block_size": input_ids.shape[1],
        "num_sequences": len(cu_seqlens) - 1,
        "total_tokens": input_ids.shape[0] * input_ids.shape[1],
    }, output_file)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="H200-Optimized Sequence Packing")

    # Input/Output
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSONL file with conversations")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for packed features")

    # Tokenizer
    parser.add_argument("--tokenizer", "-t", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Tokenizer for tokenization")

    # Processing
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="Maximum sequence length per chunk")
    parser.add_argument("--overlap_ratio", type=float, default=0.5,
                        help="Overlap ratio between chunks (0.5 = 50%%)")
    parser.add_argument("--shard_size", type=int, default=5000,
                        help="Number of blocks per output shard")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of conversations to process")

    # Output format
    parser.add_argument("--save_format", default="packed",
                        choices=["packed", "chunks"],
                        help="Output format: 'packed' (blocks) or 'chunks' (individual)")

    args = parser.parse_args()

    print("=" * 60)
    print("H200-OPTIMIZED SEQUENCE PACKING PIPELINE")
    print("=" * 60)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Tokenizer:    {args.tokenizer}")
    print(f"Max Seq Len:  {args.max_seq_len}")
    print(f"Overlap:      {args.overlap_ratio * 100:.0f}%")
    print(f"Shard Size:   {args.shard_size} blocks")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        token=hf_token,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  BOS: {tokenizer.bos_token} ({tokenizer.bos_token_id})")
    print(f"  EOS: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print(f"  PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})")

    # Initialize components
    formatter = ClaudeToQwenFormatter()
    chunker = SlidingWindowChunker(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        overlap_ratio=args.overlap_ratio
    )
    packer = BlockPacker(block_size=args.max_seq_len)

    # Load conversations
    print(f"\nLoading conversations from {args.input}...")
    conversations = load_conversations(args.input, args.limit)
    print(f"  Loaded {len(conversations):,} valid conversations")

    # Process conversations
    print("\n" + "=" * 60)
    print("STEP 1: Sliding Window Chunking with System Prompt Anchoring")
    print("=" * 60)

    all_chunks = []
    stats = {
        "total_convs": len(conversations),
        "single_chunk": 0,
        "multi_chunk": 0,
        "total_chunks": 0,
        "has_system": 0,
        "token_dist": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},  # 0:<1K, 1:1K-2K, 2:2K-3K, 3:3K-4K, 4:4K+
        "no_system": 0,
    }

    for conv in tqdm(conversations, desc="Chunking conversations"):
        chunks = chunker.chunk_conversation(
            conv["messages"],
            conv["id"],
            formatter
        )

        stats["total_chunks"] += len(chunks)
        if len(chunks) == 1:
            stats["single_chunk"] += 1
        else:
            stats["multi_chunk"] += 1

        for chunk in chunks:
            all_chunks.append(chunk)

            # Count system-prompted chunks
            if chunk.has_system:
                stats["has_system"] += 1
            else:
                stats["no_system"] += 1

            # Token distribution
            tlen = chunk.length
            if tlen < 1000:
                stats["token_dist"][0] += 1
            elif tlen < 2000:
                stats["token_dist"][1] += 1
            elif tlen < 3000:
                stats["token_dist"][2] += 1
            elif tlen < 4000:
                stats["token_dist"][3] += 1
            else:
                stats["token_dist"][4] += 1

    print(f"\nChunking Statistics:")
    print(f"  Total conversations: {stats['total_convs']:,}")
    print(f"  Total chunks:        {stats['total_chunks']:,}")
    print(f"  Single-chunk convs:  {stats['single_chunk']:,}")
    print(f"  Multi-chunk convs:   {stats['multi_chunk']:,}")
    print(f"  Chunks with system:  {stats['has_system']:,}")
    print(f"  Chunks without system: {stats['no_system']:,}")
    print(f"\n  Token length distribution:")
    dist_labels = ["<1K", "1K-2K", "2K-3K", "3K-4K", "4K+"]
    for i, label in enumerate(dist_labels):
        count = stats['token_dist'][i]
        pct = count / stats['total_chunks'] * 100 if stats['total_chunks'] > 0 else 0
        print(f"    {label:8}: {count:,} ({pct:.1f}%)")

    # Pack into blocks
    print("\n" + "=" * 60)
    print("STEP 2: Block Packing (First-Fit Decreasing)")
    print("=" * 60)

    input_ids_list, cu_seqlens_list, loss_masks_list = packer.pack(all_chunks)

    print(f"  Created {len(input_ids_list):,} blocks")

    # Calculate utilization
    total_used = sum(len(ids) for ids in input_ids_list)
    total_capacity = len(input_ids_list) * args.max_seq_len
    utilization = total_used / total_capacity * 100 if total_capacity > 0 else 0
    print(f"  Block utilization: {utilization:.1f}%")

    # Save shards
    print("\n" + "=" * 60)
    print("STEP 3: Saving Binary Tensors")
    print("=" * 60)

    num_shards = (len(input_ids_list) + args.shard_size - 1) // args.shard_size
    print(f"  Saving {len(input_ids_list):,} blocks in {num_shards} shards...")

    for shard_idx in tqdm(range(num_shards), desc="Saving shards"):
        start_idx = shard_idx * args.shard_size
        end_idx = min(start_idx + args.shard_size, len(input_ids_list))

        # Find max cu_seqlens length in this shard for padding
        shard_cu_seqlens_raw = cu_seqlens_list[start_idx:end_idx]
        max_seqlen_len = max(len(cs) for cs in shard_cu_seqlens_raw)

        # Pad cu_seqlens to same length
        padded_cu_seqlens = []
        for cs in shard_cu_seqlens_raw:
            if len(cs) < max_seqlen_len:
                # Pad with last value
                padded = torch.cat([cs, cs[-1].unsqueeze(0).expand(max_seqlen_len - len(cs))])
                padded_cu_seqlens.append(padded)
            else:
                padded_cu_seqlens.append(cs)

        # Stack blocks in this shard
        shard_input_ids = torch.stack(input_ids_list[start_idx:end_idx])
        shard_cu_seqlens = torch.stack(padded_cu_seqlens)
        shard_loss_masks = torch.stack(loss_masks_list[start_idx:end_idx])

        output_file = save_shard(
            shard_input_ids,
            shard_cu_seqlens,
            shard_loss_masks,
            output_dir,
            shard_idx
        )
        print(f"  Saved {output_file.name}: {end_idx - start_idx} blocks")

    # Save summary
    summary = {
        "input_file": str(Path(args.input).absolute()),
        "tokenizer": args.tokenizer,
        "num_conversations": stats["total_convs"],
        "num_chunks": stats["total_chunks"],
        "num_blocks": len(input_ids_list),
        "num_shards": num_shards,
        "max_seq_len": args.max_seq_len,
        "overlap_ratio": args.overlap_ratio,
        "block_utilization": utilization,
        "chunk_stats": {
            "single_chunk": stats["single_chunk"],
            "multi_chunk": stats["multi_chunk"],
            "with_system": stats["has_system"],
            "without_system": stats["no_system"],
        },
        "token_dist": stats["token_dist"],
    }

    with open(output_dir / "preprocessing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total blocks:    {len(input_ids_list):,}")
    print(f"Total shards:    {num_shards}")
    print(f"Utilization:     {utilization:.1f}%")
    print("\nNext steps:")
    print(f"  1. Extract features: python scripts/extract_features_packed.py \\")
    print(f"       --model_path <target_model> \\")
    print(f"       --input_dir {output_dir} \\")
    print(f"       --output_dir data/features_h200")
    print("=" * 60)


if __name__ == "__main__":
    main()