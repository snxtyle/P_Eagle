import argparse
import json
import os
from multiprocessing import Pool, cpu_count
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

def tokenize_conversation(args):
    """Worker function to process a single conversation string in parallel.
    Returns: (token_ids, sys_token_ids, loss_mask) where loss_mask marks conversation vs system tokens."""
    line, tokenizer_path = args
    try:
        data = json.loads(line)
        messages = data.get('messages', [])
        if not messages:
            return None

        global _tokenizer
        if '_tokenizer' not in globals():
            _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        # Full conversation tokenization
        convo_string = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids = _tokenizer.encode(convo_string, add_special_tokens=False)

        # System prompt tokenization (for loss masking)
        sys_token_ids = []
        sys_len = 0
        if messages[0]['role'] == 'system':
            sys_str = _tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=False)
            sys_token_ids = _tokenizer.encode(sys_str, add_special_tokens=False)
            sys_len = len(sys_token_ids)

        # loss_mask: 0 for system tokens (no loss), 1 for conversation tokens
        loss_mask = [0] * sys_len + [1] * (len(token_ids) - sys_len)

        return token_ids, sys_token_ids, loss_mask
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--overlap_len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Samples per output shard for feature extraction compatibility")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Initializing Multi-Core Tokenizer using {cpu_count()} workers...")

    # Step 1: Read all lines directly into RAM
    with open(args.input, 'r') as f:
        lines = f.readlines()
    if args.limit > 0:
        lines = lines[:args.limit]

    # Step 2: Parallelize Tokenization over all CPU cores
    worker_inputs = [(line, args.tokenizer) for line in lines]
    tokenized_dataset = []

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap(tokenize_conversation, worker_inputs, chunksize=64),
                           total=len(lines), desc="Parallel Tokenization (RAM Resident)"):
            if result is not None:
                tokenized_dataset.append(result)

    print(f"\nTokenized {len(tokenized_dataset)} conversations. Starting vector packing...")

    # Step 3: Fast In-Memory Sliding Window & Block Assembly with loss_mask
    all_packed = []  # List of (input_ids, loss_mask, cu_seqlens) per sequence
    current_input_ids = []
    current_loss_mask = []
    current_cu_seqlens = [0]
    accumulated_len = 0

    for token_ids, sys_token_ids, loss_mask in tokenized_dataset:
        sys_len = len(sys_token_ids)
        convo_len = len(token_ids)

        if accumulated_len + convo_len <= args.max_seq_len:
            # Fits in current block
            current_input_ids.extend(token_ids)
            current_loss_mask.extend(loss_mask)
            accumulated_len += convo_len
            current_cu_seqlens.append(accumulated_len)
        else:
            # Handle oversized conversations via sliding window
            idx = 0
            while idx < convo_len:
                remaining_space = args.max_seq_len - accumulated_len
                chunk_ids = token_ids[idx : idx + remaining_space]
                chunk_mask = loss_mask[idx : idx + remaining_space]

                current_input_ids.extend(chunk_ids)
                current_loss_mask.extend(chunk_mask)
                accumulated_len += len(chunk_ids)
                current_cu_seqlens.append(accumulated_len)

                if accumulated_len == args.max_seq_len:
                    # Finalize packed block and store as sequence
                    all_packed.append({
                        "input_ids": torch.tensor(current_input_ids, dtype=torch.long),
                        "loss_mask": torch.tensor(current_loss_mask, dtype=torch.float32),
                        "cu_seqlens": torch.tensor(current_cu_seqlens, dtype=torch.int32),
                    })

                    # Prepare next block with overlap
                    idx += remaining_space
                    idx = max(0, idx - args.overlap_len)

                    # Reset with system prompt anchor
                    current_input_ids = list(sys_token_ids)
                    current_loss_mask = [0] * sys_len
                    accumulated_len = sys_len
                    current_cu_seqlens = [0, sys_len] if sys_len > 0 else [0]
                else:
                    idx += len(chunk_ids)

    # Save any remaining data
    if current_input_ids:
        all_packed.append({
            "input_ids": torch.tensor(current_input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(current_loss_mask, dtype=torch.float32),
            "cu_seqlens": torch.tensor(current_cu_seqlens, dtype=torch.int32),
        })

    # Step 4: Save in format expected by extract_features_packed.py
    total_tokens = sum(p["input_ids"].numel() for p in all_packed)
    print(f"\nPacking complete! {len(all_packed)} blocks, {total_tokens:,} total tokens")

    # Save as shards (compatible with extract_features_packed.py)
    # NOTE: Only pad the VERY LAST block if it's incomplete. All other blocks are exactly max_seq_len.
    for i in tqdm(range(0, len(all_packed), args.shard_size), desc="Saving shards"):
        shard = all_packed[i:i + args.shard_size]

        # Check if this is the last shard (might have incomplete final block)
        is_last_shard = (i + args.shard_size >= len(all_packed))

        final_input_ids = []
        final_loss_masks = []

        for idx, s in enumerate(shard):
            ids = s["input_ids"]
            mask = s["loss_mask"]

            # Only pad the VERY LAST block if it's incomplete
            is_last_block = is_last_shard and (idx == len(shard) - 1)
            if is_last_block and len(ids) < args.max_seq_len:
                ids = F.pad(ids, (0, args.max_seq_len - len(ids)), value=0)
                mask = F.pad(mask, (0, args.max_seq_len - len(mask)), value=0.0)

            final_input_ids.append(ids)
            final_loss_masks.append(mask)

        shard_file = os.path.join(args.output, f"packed_shard_{i // args.shard_size:04d}.pt")
        torch.save({
            "input_ids": torch.stack(final_input_ids),
            "loss_mask": torch.stack(final_loss_masks),
            "cu_seqlens": [s["cu_seqlens"] for s in shard],
            "total_tokens": total_tokens,
            "num_sequences": len(shard),
        }, shard_file)

    print(f"\n✅ Processing complete!")
    print(f"   Output: {args.output}/")
    print(f"   Shards: {len(all_packed) // args.shard_size + 1}")
    print(f"   Format: Compatible with extract_features_packed.py")

if __name__ == "__main__":
    main()