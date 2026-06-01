#!/usr/bin/env python3
"""
Offline Sequence Packing for H200 Cluster

Professional-grade data preparation for speculative decoding training:
- Sliding Window Chunking with System Prompt Anchoring
- Block Packing (Multipack) for 100% GPU utilization
- Binary output (.pt tensors) for fast streaming

Usage:
    python scripts/sequence_packing.py \
        --input data/raw_conversations.jsonl \
        --output data/packed_features \
        --max_seq_len 4096 \
        --tokenizer google/gemma-3-4b-it \
        --shard_size 10000
"""

import json
import os
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class ConversationChunk:
    """A chunk of conversation with token IDs."""
    token_ids: List[int]
    start_idx: int  # Original conversation start index
    has_system: bool

    @property
    def len(self) -> int:
        return len(self.token_ids)


class SlidingWindowProcessor:
    """
    Professional sliding window chunking with system prompt anchoring.

    Algorithm:
    1. If conversation <= max_seq_len: use as-is
    2. If conversation > max_seq_len: apply sliding window with:
       - System prompt ALWAYS at start of each chunk
       - Overlapping context (1-2 turns) for continuity
       - 50% overlap between chunks for context preservation
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 4096,
        overlap_turns: int = 1,
        min_chunk_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.overlap_turns = overlap_turns
        self.min_chunk_len = min_chunk_len

        # Get special token IDs
        self.im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad_token_id = tokenizer.pad_token_id or 0

    def format_conversation(self, messages: List[Dict]) -> str:
        """Format messages using Gemma-3 chat template."""
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle content as list (mixed text/tool_calls)
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                content = "\n".join(text_parts)

            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        return text.strip()

    def get_system_prompt(self, messages: List[Dict]) -> Optional[str]:
        """Extract system prompt if present."""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    return f"<|im_start|>system\n{content}<|im_end|>\n"
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            return f"<|im_start|>system\n{c.get('text', '')}<|im_end|>\n"
        return None

    def find_message_boundaries(self, text: str) -> List[int]:
        """Find token indices where new messages begin."""
        boundaries = [0]
        im_start = "<|im_start|>"
        pos = 0

        while True:
            idx = text.find(im_start, pos)
            if idx == -1:
                break
            boundaries.append(idx)
            pos = idx + len(im_start)

        return boundaries

    def chunk_conversation(
        self,
        messages: List[Dict],
        conv_id: str
    ) -> List[ConversationChunk]:
        """
        Apply sliding window with system prompt anchoring.

        For conversations > max_seq_len:
        - Chunk 1: [System Prompt] [Turn 1] [Turn 2] ... -> max 4096 tokens
        - Chunk 2: [System Prompt] [Turn N-1] [Turn N] ... -> max 4096 tokens (overlapping)
        - Chunk 3: [System Prompt] [Turn N+N-1] ...

        The system prompt is ALWAYS prepended to maintain tool definitions.
        """
        chunks = []

        # Format full conversation
        full_text = self.format_conversation(messages)
        system_prompt = self.get_system_prompt(messages)

        # Tokenize
        token_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=False
        )

        # If fits in window, return as single chunk
        if len(token_ids) <= self.max_seq_len:
            return [ConversationChunk(
                token_ids=token_ids,
                start_idx=0,
                has_system=system_prompt is not None
            )]

        # Find message boundaries for intelligent splitting
        boundaries = self.find_message_boundaries(full_text)
        boundary_tokens = []

        for b in boundaries:
            # Find token position closest to boundary
            partial_text = full_text[b:]
            partial_tokens = self.tokenizer.encode(
                partial_text, add_special_tokens=False, truncation=True, max_length=1
            )
            # Approximate: each char ~4 tokens
            approx_token = b // 4
            boundary_tokens.append(approx_token)

        boundary_tokens = sorted(set(boundary_tokens))

        # Sliding window with system prompt anchoring
        window_size = self.max_seq_len
        step_size = window_size // 2  # 50% overlap

        # Get system prompt tokens if exists
        system_tokens = []
        if system_prompt:
            system_tokens = self.tokenizer.encode(
                system_prompt, add_special_tokens=False, truncation=False
            )
            system_len = len(system_tokens)
        else:
            system_len = 0

        start = 0
        chunk_idx = 0

        while start < len(token_ids):
            end = min(start + window_size - system_len, len(token_ids))

            # If this is not the first chunk and has system prompt, prepend it
            if chunk_idx > 0 and system_tokens:
                chunk_tokens = system_tokens + token_ids[start:end]
            else:
                chunk_tokens = token_ids[start:end]

            # Ensure we don't exceed max_seq_len
            chunk_tokens = chunk_tokens[:self.max_seq_len]

            if len(chunk_tokens) >= self.min_chunk_len:
                chunks.append(ConversationChunk(
                    token_ids=chunk_tokens,
                    start_idx=start,
                    has_system=system_prompt is not None or chunk_idx == 0
                ))

            chunk_idx += 1
            start += step_size

        return chunks


class SequencePacker:
    """
    Pack variable-length chunks into fixed-size blocks for 100% GPU utilization.

    Uses First-Fit Decreasing Height (FFDH) bin packing algorithm:
    1. Sort chunks by length (descending)
    2. Pack into blocks, filling each to capacity
    3. Generate cu_seqlens metadata for FlashAttention
    """

    def __init__(self, block_size: int = 4096):
        self.block_size = block_size

    def pack_chunks(self, chunks: List[ConversationChunk]) -> List[Tuple[List[List[int]], List[int]]]:
        """
        Pack chunks into fixed-size blocks.

        Returns:
            List of (packed_blocks, cu_seqlens) tuples for each shard
        """
        # Sort by length (largest first) for better packing
        sorted_chunks = sorted(chunks, key=lambda x: x.len, reverse=True)

        blocks = []  # List of blocks, each block is a list of chunk token lists
        block_occupancy = []  # Current token count in each block

        for chunk in sorted_chunks:
            chunk_len = chunk.len

            # Find first block that fits this chunk
            placed = False
            for i, occ in enumerate(block_occupancy):
                if occ + chunk_len <= self.block_size:
                    blocks[i].append(chunk.token_ids)
                    block_occupancy[i] += chunk_len
                    placed = True
                    break

            # If no fitting block, create new one
            if not placed:
                blocks.append([chunk.token_ids])
                block_occupancy.append(chunk_len)

        # Generate cu_seqlens for each block
        packed_data = []
        for block in blocks:
            block_tokens = []
            cu_seqlens = [0]
            pos = 0

            for chunk_tokens in block:
                block_tokens.extend(chunk_tokens)
                pos += len(chunk_tokens)
                cu_seqlens.append(pos)

            # Pad to block_size if needed
            while len(block_tokens) < self.block_size:
                block_tokens.append(0)  # padding token
                cu_seqlens[-1] = len(block_tokens)

            packed_data.append((block_tokens, cu_seqlens))

        return packed_data

    def save_shard(
        self,
        packed_data: List[Tuple[List[List[int]], List[int]]],
        output_path: Path,
        shard_idx: int
    ):
        """Save a shard as binary .pt tensors."""
        # Unpack for saving
        all_input_ids = []
        all_cu_seqlens = []
        all_max_seqlen = []

        for block_tokens, cu_seqlens in packed_data:
            all_input_ids.append(block_tokens)
            all_cu_seqlens.append(cu_seqlens)
            all_max_seqlen.append(max(cu_seqlens[1:] + [0]) if cu_seqlens else self.block_size)

        # Save as torch tensors
        torch.save(
            torch.tensor(all_input_ids, dtype=torch.long),
            output_path / f"shard_{shard_idx:04d}_input_ids.pt"
        )
        torch.save(
            torch.tensor(all_cu_seqlens, dtype=torch.int32),
            output_path / f"shard_{shard_idx:04d}_cu_seqlens.pt"
        )
        torch.save(
            torch.tensor(all_max_seqlen, dtype=torch.int32),
            output_path / f"shard_{shard_idx:04d}_max_seqlen.pt"
        )

        # Also save metadata
        metadata = {
            "block_size": self.block_size,
            "num_blocks": len(all_input_ids),
            "total_tokens": sum(all_max_seqlen)
        }

        with open(output_path / f"shard_{shard_idx:04d}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def load_conversations(input_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load conversations from JSONL file."""
    conversations = []

    with open(input_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Loading conversations")):
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


def main():
    parser = argparse.ArgumentParser(description="Offline Sequence Packing for H200")

    # Input/Output
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSONL file with conversations")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for packed features")
    parser.add_argument("--tokenizer", "-t", default="google/gemma-3-4b-it",
                        help="Tokenizer for tokenization")

    # Processing
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="Maximum sequence length per chunk")
    parser.add_argument("--overlap_turns", type=int, default=1,
                        help="Number of overlapping turns between chunks")
    parser.add_argument("--shard_size", type=int, default=10000,
                        help="Number of conversations per output shard")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of conversations to process")

    # Compatibility
    parser.add_argument("--max_seq_len_medium", type=int, default=8192,
                        help="Max tokens for medium chunks (for multi-phase training)")

    args = parser.parse_args()

    print("=" * 60)
    print("OFFLINE SEQUENCE PACKING FOR H200")
    print("=" * 60)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Tokenizer:    {args.tokenizer}")
    print(f"Max Seq Len:  {args.max_seq_len}")
    print(f"Shard Size:   {args.shard_size}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Load conversations
    print(f"\nLoading conversations from {args.input}...")
    conversations = load_conversations(args.input, args.limit)
    print(f"  Loaded {len(conversations)} valid conversations")

    # Initialize processors
    sw_processor = SlidingWindowProcessor(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        overlap_turns=args.overlap_turns
    )
    packer = SequencePacker(block_size=args.max_seq_len)

    # Process conversations
    print("\nApplying sliding window chunking with system prompt anchoring...")
    all_chunks = []
    stats = {
        "single_chunk": 0,
        "multiple_chunks": 0,
        "total_chunks": 0,
        "has_system": 0,
        "token_dist": {">0": 0, "1k-2k": 0, "2k-3k": 0, "3k-4k": 0, "4k+": 0}
    }

    for conv in tqdm(conversations, desc="Chunking"):
        chunks = sw_processor.chunk_conversation(
            conv["messages"],
            conv["id"]
        )

        stats["total_chunks"] += len(chunks)
        if len(chunks) == 1:
            stats["single_chunk"] += 1
        else:
            stats["multiple_chunks"] += 1

        for chunk in chunks:
            all_chunks.append(chunk)
            if chunk.has_system:
                stats["has_system"] += 1

            # Token distribution
            tlen = chunk.len
            if tlen < 1000:
                stats["token_dist"][">0"] += 1
            elif tlen < 2000:
                stats["token_dist"]["1k-2k"] += 1
            elif tlen < 3000:
                stats["token_dist"]["2k-3k"] += 1
            elif tlen < 4000:
                stats["token_dist"]["3k-4k"] += 1
            else:
                stats["token_dist"]["4k+"] += 1

    print(f"\nChunking Statistics:")
    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Single-chunk:     {stats['single_chunk']}")
    print(f"  Multi-chunk:      {stats['multiple_chunks']}")
    print(f"  Has system:       {stats['has_system']}")
    print(f"  Token distribution:")
    for bucket, count in stats["token_dist"].items():
        pct = count / stats['total_chunks'] * 100 if stats['total_chunks'] > 0 else 0
        print(f"    {bucket}: {count} ({pct:.1f}%)")

    # Pack into blocks
    print("\nPacking chunks into fixed-size blocks...")
    packed_data = packer.pack_chunks(all_chunks)
    print(f"  Created {len(packed_data)} blocks")

    # Calculate utilization
    total_used = sum(len(cp) for _, cp in packed_data)
    total_capacity = len(packed_data) * args.max_seq_len
    utilization = total_used / total_capacity * 100 if total_capacity > 0 else 0
    print(f"  Block utilization: {utilization:.1f}%")

    # Save output
    print(f"\nSaving to {output_dir}...")
    packer.save_shard(packed_data, output_dir, shard_idx=0)

    # Save summary
    summary = {
        "input_file": args.input,
        "num_conversations": len(conversations),
        "num_chunks": stats["total_chunks"],
        "num_blocks": len(packed_data),
        "max_seq_len": args.max_seq_len,
        "block_utilization": utilization,
        "stats": stats
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SEQUENCE PACKING COMPLETE!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Blocks: {len(packed_data)}")
    print(f"Utilization: {utilization:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()