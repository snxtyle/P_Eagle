#!/usr/bin/env python3
"""
P-EAGLE Feature Extraction Script

Extracts tri-layer fused hidden states from Target Model for training the Drafter.
"""

import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from pathlib import Path

# Load HF token from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

from ..utils.feature_utils import EagleDataset, align_segments_to_tokens, fuse_tri_layer_features


class TriLayerConfig:
    """Configuration for which layers to extract from."""

    def __init__(self, model, mode: str = "early,middle,final"):
        # Handle different config structures (Gemma 2, Gemma 3, etc.)
        if hasattr(model.config, 'num_hidden_layers'):
            self.num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_hidden_layers'):
            # Gemma 3 and multimodal models have nested text_config
            self.num_layers = model.config.text_config.num_hidden_layers
        else:
            raise AttributeError(f"Cannot determine num_hidden_layers from model config. "
                               f"Available attrs: {dir(model.config)}")
        self.mode = mode

        if mode == "last":
            # Use the last transformer layer output directly (matches lm_head training)
            # hidden_states tuple: [0]=embeddings, [1]=layer0, ..., [-1]=last_layer
            self.layer_indices = [-1]
            print(f"Feature extraction mode: LAST LAYER (index -1, layer {self.num_layers - 1})")
            print("  This matches the lm_head training distribution.")
        elif mode == "early,middle,final":
            self.layer_indices = [
                self.num_layers // 4,
                self.num_layers // 2,
                3 * self.num_layers // 4
            ]
            print(f"Tri-Layer Config: extracting from layers {self.layer_indices} (MEAN fusion)")
            print("  WARNING: Fused features may not align with lm_head!")
        elif mode == "first,middle,last":
            self.layer_indices = [0, self.num_layers // 2, self.num_layers - 1]
            print(f"Tri-Layer Config: extracting from layers {self.layer_indices}")
        elif mode == "all":
            self.layer_indices = list(range(self.num_layers))
            print(f"Feature extraction: ALL layers {self.layer_indices}")
        else:
            self.layer_indices = [int(x) for x in mode.split(",")]
            print(f"Feature extraction: custom layers {self.layer_indices}")


class FeatureExtractor:
    """Main feature extraction pipeline for P-EAGLE."""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        tokenizer_name: str = None,
        quantization: str = "8bit",
        layer_config: str = "last",  # Last layer only (required for speculative decoding alignment)
        fusion_mode: str = "mean",
        max_length: int = 4096,  # Increased for H200 141GB VRAM
        batch_size: int = 1,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fusion_mode = fusion_mode
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        # Setup quantization
        self.quant_config = self._setup_quantization(quantization)

        # Load model and tokenizer with HF token for gated models
        print(f"Loading target model: {model_name}")
        if HF_TOKEN:
            print("Using HF token from .env file")

        # Use drafter's tokenizer to ensure compatibility during training
        tokenizer_to_use = tokenizer_name if tokenizer_name else model_name
        print(f"Loading tokenizer: {tokenizer_to_use}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_to_use, token=HF_TOKEN)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check vocabulary compatibility if using different models
        if tokenizer_name and tokenizer_name != model_name:
            self._check_vocab_compatibility(model_name, tokenizer_to_use)

        print("Loading model (this may take a few minutes)...")
        print("Using synchronous loading to avoid watchdog detection...")

        # CRITICAL FIX: Force synchronous CUDA operations
        # This prevents async transfer pattern detection by watchdog
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.synchronize()

        # Prepare loading kwargs - dtype only for non-quantized models
        load_kwargs = {
            "low_cpu_mem_usage": True,  # Stream directly to GPU
            "token": HF_TOKEN,
        }
        if quantization != "none":
            load_kwargs["quantization_config"] = self.quant_config
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        print("Model loaded to CPU")

        # Force sync before eval
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.model.eval()
        print("eval() complete")

        # Move to CUDA synchronously
        if quantization == "none" and torch.cuda.is_available():
            print("Moving model to CUDA (synchronous)...")
            with torch.no_grad():
                self.model = self.model.to("cuda:0")
            torch.cuda.synchronize()  # Force completion
            print("Moved to CUDA")

        device = next(self.model.parameters()).device
        print(f"Model loaded on device: {device}")

        print("Initializing layer config...")
        self.layer_config = TriLayerConfig(self.model, layer_config)
        print(f"Layer config complete: layers {self.layer_config.layer_indices}")
        # Handle different config structures for hidden_size
        if hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'hidden_size'):
            hidden_size = self.model.config.text_config.hidden_size
        else:
            hidden_size = "unknown"
        print(f"Model loaded. Hidden dim: {hidden_size}")
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")

        # OPTIMIZATION: Compile model for faster inference
        if hasattr(torch, 'compile') and quantization == "none":
            print("Compiling model with torch.compile() for faster inference...")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled successfully")
            except Exception as e:
                print(f"Could not compile model: {e}")

    def _setup_quantization(self, mode: str):
        if mode == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _check_vocab_compatibility(self, target_model: str, drafter_model: str):
        """Check if target and drafter have compatible vocabularies."""
        print("\n" + "="*60)
        print("Checking vocabulary compatibility...")
        print(f"  Target:  {target_model}")
        print(f"  Drafter: {drafter_model}")

        try:
            tok_target = AutoTokenizer.from_pretrained(target_model, token=HF_TOKEN)
            tok_drafter = AutoTokenizer.from_pretrained(drafter_model, token=HF_TOKEN)

            vocab_target = len(tok_target)
            vocab_drafter = len(tok_drafter)

            print(f"\n  Target vocab size:  {vocab_target}")
            print(f"  Drafter vocab size: {vocab_drafter}")

            # Test encoding with sample texts
            test_texts = ["Hello world", "The quick brown fox", "Payment processing"]
            mismatches = 0

            for text in test_texts:
                enc_target = tok_target.encode(text, add_special_tokens=False)
                enc_drafter = tok_drafter.encode(text, add_special_tokens=False)
                if enc_target != enc_drafter:
                    mismatches += 1

            if mismatches == 0:
                print(f"  Tokenization: IDENTICAL (all {len(test_texts)} test strings match)")
                print("="*60)
            else:
                print(f"  WARNING: {mismatches}/{len(test_texts)} test strings differ!")
                print("  Models may produce gibberish output due to vocab mismatch.")
                print("  Recommendation: Use models from the same family (e.g., both Gemma).")
                print("="*60)

        except Exception as e:
            print(f"  Could not check compatibility: {e}")
            print("="*60)

    @torch.no_grad()
    def extract_sample(self, samples):
        """Extract features from samples.
        _extract_single now returns a list (one per chunk), so flatten here.
        """
        results = []
        for sample in samples:
            try:
                chunk_results = self._extract_single(sample)
                if chunk_results is not None:
                    results.extend(chunk_results)
            except Exception as e:
                print(f"Error extracting sample: {e}")
        return results

    def _parse_content(self, content):
        """Parse content that may be string-encoded JSON list or null.

        Handles multiple formats:
        - Plain string: "Hello world"
        - List of dicts: [{"type": "text", "text": "Hello"}]
        - String-encoded list: "[{'type': 'text', 'text': 'Hello'}]"
        """
        if content is None:
            return ""
        if isinstance(content, list):
            # Handle actual list format - extract text from text blocks
            texts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text" and "text" in c:
                        texts.append(c["text"])
                    elif "text" in c:
                        texts.append(c["text"])
                    elif "content" in c:
                        texts.append(c["content"])
            return " ".join(texts) if texts else ""
        if isinstance(content, str):
            # Check if it's a string-encoded JSON/Python list
            stripped = content.strip()
            if stripped.startswith("[") and ("'type'" in stripped or '"type"' in stripped):
                try:
                    # Try to parse as Python literal (handles single quotes)
                    import ast
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, list):
                        texts = []
                        for c in parsed:
                            if isinstance(c, dict):
                                if c.get("type") == "text" and "text" in c:
                                    texts.append(c["text"])
                                elif "text" in c:
                                    texts.append(c["text"])
                        return " ".join(texts) if texts else ""
                except (ValueError, SyntaxError):
                    # Fallback: try JSON with quote replacement
                    try:
                        import json
                        normalized = stripped.replace("'", '"')
                        parsed = json.loads(normalized)
                        if isinstance(parsed, list):
                            texts = [c.get("text", "") for c in parsed if isinstance(c, dict)]
                            return " ".join(texts)
                    except:
                        pass
            return content
        return str(content)

    @torch.no_grad()
    def _extract_single(self, batch_item):
        """Extract features from a single sample.
        
        Uses sliding window chunks for conversations longer than max_length.
        Each chunk preserves the causal prefix but only saves the window portion.
        Hard-caps at 131072 tokens to prevent OOM on extreme outliers.
        """
        try:
            conversation_text = batch_item["conversation_text"]
            messages = batch_item["original_messages"]

            for msg in messages:
                if "content" in msg:
                    msg["content"] = self._parse_content(msg.get("content"))

            segments = batch_item.get("segments", [])
            if not segments and messages:
                assistant_roles = {"assistant", "bot", "ai"}
                segments = []
                assistant_count = 0
                for i, msg in enumerate(messages):
                    role = msg.get("role", "unknown").lower().strip()
                    content = msg.get("content", "")
                    is_assistant = role in assistant_roles
                    if is_assistant and not content:
                        mask = 0
                    else:
                        mask = 1 if is_assistant else 0
                        if mask:
                            assistant_count += 1
                    segments.append({"index": i, "role": role, "mask": mask})

                if assistant_count == 0:
                    print(f"  CRITICAL: No assistant messages found. Roles: {[m.get('role') for m in messages]}")

            # Tokenize full conversation (no truncation)
            inputs = self.tokenizer(
                conversation_text,
                return_tensors="pt",
                truncation=False,
                padding=True
            )
            full_input_ids = inputs["input_ids"].to(self.model.device)
            full_attention = inputs["attention_mask"].to(self.model.device)
            full_seq_len = full_input_ids.shape[1]

            # Hard cap at 131072 to prevent OOM on extreme outliers (3.4M word samples)
            max_input_tokens = 131072
            if full_seq_len > max_input_tokens:
                print(f"  Truncating {full_seq_len} → {max_input_tokens} tokens (hard cap)")
                full_input_ids = full_input_ids[:, :max_input_tokens]
                full_attention = full_attention[:, :max_input_tokens]
                full_seq_len = max_input_tokens

            # Compute full segment mask once (reused across all chunks)
            full_token_mask = align_segments_to_tokens(
                messages, segments, self.tokenizer, full_input_ids[0]
            )
            if full_token_mask.sum().item() > 0:
                print(f"  Mask covers {full_token_mask.sum().item()} trainable tokens")

            def _make_chunk_result(hidden, raw_hn, inp_ids, attn_mask, tgt_ids, mask_slice):
                return {
                    "text": conversation_text,
                    "input_ids": inp_ids.cpu(),
                    "fused_hidden_states": hidden.cpu(),
                    "raw_hidden_states": raw_hn.cpu(),
                    "norm_mean": None,
                    "norm_std": None,
                    "loss_mask": mask_slice.cpu(),
                    "attention_mask": attn_mask.cpu(),
                    "target_token_ids": tgt_ids.cpu(),
                }

            def _norm_last_layer(hs):
                # Gemma 3 multimodal: model.language_model.norm
                if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'norm'):
                    return self.model.language_model.norm(hs)
                # Standard: model.model.norm (e.g., Llama, Mistral)
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                    return self.model.model.norm(hs)
                # Some models: model.norm
                elif hasattr(self.model, 'norm'):
                    return self.model.norm(hs)
                # Gemma 2: model.model.language_model.norm
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'norm'):
                    return self.model.model.language_model.norm(hs)
                # Fallback: return as-is (no normalization)
                # WARNING: This may cause training issues - ideally the model should have a final norm
                return hs

            if full_seq_len <= self.max_length:
                print(f"  Processing {full_seq_len} tokens...")
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention,
                    output_hidden_states=True
                )
                target_ids = outputs.logits.argmax(dim=-1)[0]
                hidden = _norm_last_layer(outputs.hidden_states[-1][0])

                if full_token_mask.sum().item() == 0:
                    print("  WARNING: Empty loss mask")

                return [_make_chunk_result(
                    hidden, hidden,
                    full_input_ids[0], full_attention[0],
                    target_ids, full_token_mask
                )]

            # Standalone overlapping chunks — each chunk is processed independently
            # at exactly max_length tokens (no prefix accumulation, avoids OOM)
            chunk_size = self.max_length
            overlap = min(8192, chunk_size // 4)
            stride = chunk_size - overlap

            results = []
            for start in range(0, full_seq_len, stride):
                end = min(start + chunk_size, full_seq_len)
                window_len = end - start
                if window_len < min(4096, chunk_size // 4):
                    break

                chunk_input = full_input_ids[:, start:end]
                chunk_attn = full_attention[:, start:end]

                print(f"  Chunk [{start}:{end}] ({window_len} tokens) of {full_seq_len}...")
                outputs = self.model(
                    input_ids=chunk_input,
                    attention_mask=chunk_attn,
                    output_hidden_states=True
                )

                hidden = _norm_last_layer(outputs.hidden_states[-1][0])
                target_ids = outputs.logits.argmax(dim=-1)[0]
                mask_slice = full_token_mask[start:end]

                results.append(_make_chunk_result(
                    hidden, hidden,
                    full_input_ids[0, start:end], full_attention[0, start:end],
                    target_ids, mask_slice
                ))

                del outputs
                torch.cuda.empty_cache()

            print(f"  → {len(results)} chunks from {full_seq_len} tokens")
            return results

        except Exception as e:
            print(f"Error extracting sample: {e}")
            import traceback
            traceback.print_exc()
            return None

    @torch.no_grad()
    def _extract_batch_optimized(self, batch_items):
        """Extract features from a batch of samples efficiently.

        OPTIMIZATION: Processes all samples in a batch together by padding
        to the same length and running the model once.
        """
        if not batch_items:
            return []

        try:
            # Process all texts
            all_texts = []
            all_messages = []
            all_segments = []

            for item in batch_items:
                conversation_text = item["conversation_text"]
                messages = item["original_messages"]

                # Parse complex content formats
                for msg in messages:
                    if "content" in msg:
                        msg["content"] = self._parse_content(msg.get("content"))

                # Auto-generate segments
                segments = item.get("segments", [])
                if not segments and messages:
                    assistant_roles = {"assistant", "bot", "ai"}
                    segments = []
                    for i, msg in enumerate(messages):
                        role = msg.get("role", "unknown").lower().strip()
                        content = msg.get("content", "")
                        is_assistant = role in assistant_roles
                        mask = 0 if (is_assistant and not content) else (1 if is_assistant else 0)
                        segments.append({"index": i, "role": role, "mask": mask})

                all_texts.append(conversation_text)
                all_messages.append(messages)
                all_segments.append(segments)

            # Tokenize as a batch (with padding)
            inputs = self.tokenizer(
                all_texts,
                return_tensors="pt",
                truncation=False,
                padding=True  # Pad to longest in batch
            )

            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            # Run model ONCE for the entire batch
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            all_hidden_states = outputs.hidden_states

            # CRITICAL FIX: Save actual target token IDs from model's own logits
            target_token_ids = outputs.logits.argmax(dim=-1)  # [batch, seq_len]

            # Process each sample in the batch
            results = []
            for i in range(len(batch_items)):
                # Get valid length for this sample (excluding padding)
                valid_len = attention_mask[i].sum().item()

                # Extract features for this sample
                sample_input_ids = input_ids[i, :valid_len]
                sample_attention = attention_mask[i, :valid_len]

                if self.layer_config.mode == "last":
                    last_hidden = all_hidden_states[-1]
                    fused_hidden = last_hidden[i:i+1, :valid_len, :]
                    raw_hidden = last_hidden[i:i+1, :valid_len, :]
                    mean_stats = None
                    std_stats = None
                else:
                    fusion_result = fuse_tri_layer_features(
                        [hs[i:i+1, :valid_len, :] for hs in all_hidden_states],
                        self.layer_config.layer_indices,
                        self.fusion_mode
                    )
                    fused_hidden = fusion_result['fused']
                    raw_hidden = fusion_result['raw']
                    mean_stats = fusion_result['mean']
                    std_stats = fusion_result['std']

                # Align segments
                token_level_mask = align_segments_to_tokens(
                    all_messages[i],
                    all_segments[i],
                    self.tokenizer,
                    sample_input_ids
                )

                results.append({
                    "text": all_texts[i],
                    "input_ids": sample_input_ids.cpu(),
                    "fused_hidden_states": fused_hidden[0].cpu(),
                    "raw_hidden_states": raw_hidden[0].cpu(),
                    "norm_mean": mean_stats[0].cpu() if mean_stats is not None else None,
                    "norm_std": std_stats[0].cpu() if std_stats is not None else None,
                    "loss_mask": token_level_mask.cpu(),
                    "attention_mask": sample_attention.cpu(),
                    "target_token_ids": target_token_ids[i, :valid_len].cpu()
                })

            return results

        except Exception as e:
            print(f"Error in batch extraction: {e}")
            # Fallback to single processing
            return [self._extract_single(item) for item in batch_items]

    def process_file(self, input_path: str, shard_size: int = 1000):
        dataset = EagleDataset(input_path, self.tokenizer, self.max_length)

        # Custom collate that returns list (variable-length sequences)
        def collate_fn(batch):
            return batch

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            prefetch_factor=2
        )

        all_features = []
        shard_idx = 0

        for batch in tqdm(dataloader, desc="Extracting features"):
            batch_results = self.extract_sample(batch)
            all_features.extend(batch_results)

            if len(all_features) >= shard_size:
                self._save_shard(all_features, input_path, shard_idx)
                all_features = []
                shard_idx += 1

        if all_features:
            self._save_shard(all_features, input_path, shard_idx)

        print(f"Extraction complete. Saved {shard_idx + 1} shards to {self.output_dir}")

    def _save_shard(self, features, input_path, shard_idx):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        # Convert to bf16 to reduce shard size by ~50%
        hidden_states = torch.nn.utils.rnn.pad_sequence(
            [f["fused_hidden_states"] for f in features],
            batch_first=True,
            padding_value=0.0
        ).to(torch.bfloat16)
        # Save raw hidden states for lm_head compatibility
        raw_hidden_states = torch.nn.utils.rnn.pad_sequence(
            [f["raw_hidden_states"] for f in features],
            batch_first=True,
            padding_value=0.0
        ).to(torch.bfloat16)
        # Save normalization statistics for denormalization
        norm_means = None
        norm_stds = None
        if features[0].get("norm_mean") is not None:
            norm_means = torch.nn.utils.rnn.pad_sequence(
                [f["norm_mean"] for f in features],
                batch_first=True,
                padding_value=0.0
            )
            norm_stds = torch.nn.utils.rnn.pad_sequence(
                [f["norm_std"] for f in features],
                batch_first=True,
                padding_value=1.0  # Default std is 1
            )
        loss_masks = torch.nn.utils.rnn.pad_sequence(
            [f["loss_mask"] for f in features],
            batch_first=True,
            padding_value=0
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0
        )
        # CRITICAL FIX: Save actual target token IDs from model's own logits.
        # These are precomputed next-token predictions (argmax of outputs.logits).
        # Using them for CE loss ensures training targets are correct regardless
        # of which layer(s) we extract for hidden-state MSE loss.
        target_token_ids = torch.nn.utils.rnn.pad_sequence(
            [f["target_token_ids"] for f in features],
            batch_first=True,
            padding_value=-100
        )

        # Extract text for flexible retokenization with any drafter
        texts = [f["text"] for f in features]

        input_name = Path(input_path).stem
        output_file = self.output_dir / f"{input_name}_shard{shard_idx:04d}.pt"

        # Save target model's lm_head weights for perfect KL alignment during training
        # CRITICAL FIX: We need to try multiple locations and handle errors
        lm_head_state = None
        lm_head_found = False

        # Try multiple locations where lm_head might be stored
        # IMPORTANT: Gemma-3 multimodal models (like gemma-3-4b-it) use model.model.language_model.lm_head
        locations_to_check = [
            ("self.model.lm_head", lambda: self.model if hasattr(self.model, 'lm_head') and self.model.lm_head is not None else None),
            ("self.model.model.lm_head", lambda: self.model.model if hasattr(self.model, 'model') and hasattr(self.model.model, 'lm_head') and self.model.model.lm_head is not None else None),
            ("self.model.model.model.lm_head", lambda: self.model.model.model if hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'lm_head') and self.model.model.model.lm_head is not None else None),
            ("self.model.language_model.lm_head", lambda: self.model.language_model if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'lm_head') and self.model.language_model.lm_head is not None else None),
            # CRITICAL: Gemma-3 multimodal models (gemma-3-4b-it) structure
            ("self.model.model.language_model.lm_head", lambda: self.model.model.language_model if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'lm_head') and self.model.model.language_model.lm_head is not None else None),
        ]

        for location_name, check_func in locations_to_check:
            try:
                lm_obj = check_func()
                if lm_obj is not None:
                    lm_head_state = lm_obj.state_dict()
                    if lm_head_state is not None:
                        weight_shape = lm_head_state.get('weight', torch.tensor([])).shape
                        print(f"  ✓ Saved lm_head from {location_name} (shape: {weight_shape})")
                        lm_head_found = True
                        break
            except Exception as e:
                print(f"  Note: Could not access {location_name}: {e}")

        if not lm_head_found:
            print(f"  ⚠️  WARNING: Could not find lm_head in model - training will use random initialization!")
            print(f"  This will cause poor speculative decoding performance.")
            print(f"  Model type: {type(self.model).__name__}")
            print(f"  Model attributes with 'lm': {[attr for attr in dir(self.model) if 'lm' in attr.lower()]}")

        save_dict = {
            "texts": texts,
            "input_ids": input_ids,
            "fused_hidden_states": hidden_states,
            "raw_hidden_states": raw_hidden_states,
            "loss_mask": loss_masks,
            "attention_mask": attention_masks,
            "target_token_ids": target_token_ids,
            "model_name": self.model_name,
            "layer_indices": self.layer_config.layer_indices,
            "fusion_mode": self.fusion_mode,
            "num_samples": len(features),
            "lm_head": lm_head_state,
            "vocab_size": len(self.tokenizer),
            "hidden_size": getattr(self.model.config, 'hidden_size',
                                   getattr(getattr(self.model.config, 'text_config', None), 'hidden_size', 0))
        }
        # Add normalization stats if available
        if norm_means is not None:
            save_dict["norm_mean"] = norm_means
            save_dict["norm_std"] = norm_stds

        torch.save(save_dict, output_file)

        print(f"Saved {output_file} ({len(features)} samples)")


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Feature Extraction")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None, help="Tokenizer to use (defaults to model_path). Use drafter model for compatibility.")
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_dir", default="./features")
    parser.add_argument("--quantization", default="8bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--layers", default="last",
                        help="Which layers to extract: 'last' (default — required for speculative decoding), 'early,middle,final', 'first,middle,last', or comma-separated indices")
    parser.add_argument("--fusion", default="mean", choices=["mean", "weighted", "concat", "none"],
                        help="How to fuse multiple layers: 'mean' (default), 'weighted', 'concat', 'none'")
    parser.add_argument("--max_length", type=int, default=65535,
                        help="Max sequence length (default: 65535 — effectively no truncation)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=50,
                        help="Samples per shard (default: 50 for ~2GB shards)")

    args = parser.parse_args()

    extractor = FeatureExtractor(
        model_name=args.model_path,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_path,
        quantization=args.quantization,
        layer_config=args.layers,
        fusion_mode=args.fusion,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    extractor.process_file(args.input_data, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
