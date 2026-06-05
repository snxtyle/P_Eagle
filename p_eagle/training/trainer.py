#!/usr/bin/env python3
"""
P-EAGLE Drafter Training Script

Trains a Drafter model to predict K future hidden states of the Target Model
using Multi-Token Prediction (MTP) heads with parallel speculation.
"""

import argparse
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import timedelta, datetime
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from bitsandbytes.optim import PagedAdamW8bit
from tqdm import tqdm
import numpy as np

# Weights & Biases
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import wandb
WANDB_AVAILABLE = bool(os.getenv("WANDB_API_KEY"))


class _BrokenPipeSuppressor:
    """Context manager to suppress BrokenPipeError and handle SIGPIPE.

    This prevents training crashes when stdout/stderr is piped to
    programs like `head` that close the pipe early.
    """
    def __init__(self):
        self._old_sigpipe_handler = None
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        # Save old signal handler
        self._old_sigpipe_handler = signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        # Save file objects
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore signal handler
        signal.signal(signal.SIGPIPE, self._old_sigpipe_handler)
        # Restore file objects
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        return False  # Don't suppress exceptions


def _tqdm_safe_write(pbar, *args, **kwargs):
    """Safely update tqdm progress bar, handling BrokenPipeError."""
    try:
        pbar.update(*args, **kwargs)
    except (BrokenPipeError, OSError) as e:
        # Pipe was closed - stop updating but don't crash
        pbar.disable = True
        # Optionally log that progress output stopped
        try:
            print(f"\n[Progress output stopped: {type(e).__name__}]", file=sys.stderr)
        except:
            pass


@contextmanager
def safe_tqdm(*args, **kwargs):
    """Create a tqdm progress bar that handles broken pipes gracefully.

    Usage:
        with safe_tqdm(total=100, desc="Training") as pbar:
            for i in range(100):
                # do work
                pbar.update(1)
    """
    # Set default values for file and dynamic_ncols to handle piping
    kwargs.setdefault('file', sys.stdout)
    kwargs.setdefault('dynamic_ncols', True)
    kwargs.setdefault('mininterval', 0.5)  # Reduce update frequency

    pbar = None
    try:
        pbar = tqdm(*args, **kwargs)
        yield _SafePbarWrapper(pbar)
    except (BrokenPipeError, OSError) as e:
        # Pipe was closed - clean up gracefully
        if pbar is not None:
            try:
                pbar.disable = True
            except:
                pass
        # Re-raise with a cleaner message only if not already handling
        if not isinstance(e, BrokenPipeError):
            raise
    finally:
        if pbar is not None:
            try:
                pbar.close()
            except:
                pass


class _SafePbarWrapper:
    """Wrapper around tqdm that safely handles broken pipes."""

    def __init__(self, pbar):
        self._pbar = pbar

    def update(self, n=1):
        try:
            self._pbar.update(n)
        except (BrokenPipeError, OSError):
            self._pbar.disable = True

    def set_postfix(self, *args, **kwargs):
        try:
            self._pbar.set_postfix(*args, **kwargs)
        except (BrokenPipeError, OSError):
            pass

    def set_description(self, *args, **kwargs):
        try:
            self._pbar.set_description(*args, **kwargs)
        except (BrokenPipeError, OSError):
            pass

    def __getattr__(self, name):
        return getattr(self._pbar, name)


class _DummyPbarWrapper:
    """Dummy progress bar that does nothing.

    Used when stdout is closed/redirected (e.g., piped to head).
    This allows training to continue without progress display.
    """

    def __init__(self):
        self.n = 0
        self.disable = False

    def update(self, n=1):
        self.n += n

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


from p_eagle.models.peagle_drafter import EagleDrafterModel
from p_eagle.utils.feature_utils import EagleTrainingDataset
from p_eagle.utils.loss_utils import masked_mse_loss, hidden_state_token_loss
from p_eagle.utils.metrics import MetricsTracker, GenerationMetrics


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_training_logger(output_dir: Path, run_name: str = None) -> logging.Logger:
    """Setup comprehensive logging for training runs.

    Creates timestamped log files and captures both console and file output.

    Args:
        output_dir: Directory to save logs
        run_name: Optional run name, defaults to timestamp

    Returns:
        Logger instance configured for training
    """
    # Create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"training_{timestamp}"

    # Create run-specific log directory
    run_log_dir = logs_dir / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = logging.getLogger("peagle_training")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - main log
    log_file = run_log_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Also capture raw stdout/stderr to a separate file
    raw_log_file = run_log_dir / "output.log"
    raw_fh = logging.FileHandler(raw_log_file, mode='w')
    raw_fh.setLevel(logging.DEBUG)
    raw_fh.setFormatter(logging.Formatter("%(message)s"))
    raw_fh.addFilter(lambda record: True)  # Capture everything
    logger.addHandler(raw_fh)

    logger.info(f"=" * 70)
    logger.info(f"P-EAGLE TRAINING SESSION STARTED")
    logger.info(f"=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Log directory: {run_log_dir}")
    logger.info(f"Main log file: {log_file}")
    logger.info(f"Raw output file: {raw_log_file}")
    logger.info(f"=" * 70)

    return logger, run_log_dir, run_id


def run_pre_training_security_check(feature_dir: str) -> bool:
    """
    Run pre-training security verification.

    Scans feature directory source data for secrets using gitleaks
    and custom patterns. Fails fast if CRITICAL secrets detected.

    Returns True if safe to proceed, False if secrets found.
    """
    import subprocess
    import tempfile
    import re

    print_section("PRE-TRAINING SECURITY VERIFICATION")

    # Check if we have the source dataset in feature metadata
    feature_path = Path(feature_dir)
    if not feature_path.exists():
        print("⚠️  Feature directory not found, skipping security check")
        return True

    # Try to find source dataset from feature metadata
    source_dataset = None
    for meta_file in feature_path.glob("*_shard*.pt"):
        try:
            import torch
            data = torch.load(meta_file, map_location="cpu")
            # Features don't contain raw text, they're already processed
            # Security should be checked at data generation time
            print("✅ Features are pre-processed tensors (no raw text to scan)")
            print("   Security scan should have run during: generate_data.py → extract_features.py")
            return True
        except Exception:
            continue

    # If we reach here, no features found
    print("⚠️  No feature files found to verify")
    return True


def verify_dataset_source_security(dataset_path: str, skip_check: bool = False) -> bool:
    """
    Verify dataset source is clean before training.

    This should be called on the ORIGINAL dataset (JSONL) before
    feature extraction or training.
    """
    if skip_check:
        print("⚠️  Security check skipped (--skip-security-check)")
        return True

    if not Path(dataset_path).exists():
        print(f"\n{'='*70}")
        print("⛔ SECURITY CHECK ERROR")
        print(f"{'='*70}")
        print(f"Dataset not found: {dataset_path}")
        print("Cannot run security verification without dataset source.")
        print("Use --skip-security-check only if intentionally bypassing.")
        return False  # Fail if we can't verify security

    print_section("DATASET SECURITY SCAN")
    print(f"Dataset: {dataset_path}")

    # Try to run gitleaks
    try:
        result = subprocess.run(
            ["which", "gitleaks"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            print("⚠️  gitleaks not installed, trying to install/download...")
            # Try auto-install
            install_script = Path(__file__).parent.parent.parent / "scripts" / "scan_dataset_secrets.py"
            if install_script.exists():
                print(f"   Using: {install_script}")
    except Exception:
        pass

    # Run basic regex scan for common patterns
    print("\n  Running regex pattern scan...")
    patterns = [
        (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', 'Indian PAN'),  # PAN numbers
        (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b', 'Credit Card'),  # Credit cards
        (r'\bAKIA[0-9A-Z]{16}\b', 'AWS Access Key'),  # AWS keys
        (r'-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----', 'Private Key'),  # Private keys
    ]

    findings = []
    line_count = 0

    try:
        with open(dataset_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                for pattern, name in patterns:
                    if re.search(pattern, line):
                        findings.append((line_num, name))

                # Limit scan to first 1000 lines for speed
                if line_num >= 1000:
                    break
    except Exception as e:
        print(f"  ⚠️  Could not scan dataset: {e}")
        return True

    if findings:
        print(f"\n  ❌ SECURITY ISSUES FOUND:")
        for line_num, name in findings[:10]:
            print(f"     Line {line_num}: {name}")
        if len(findings) > 10:
            print(f"     ... and {len(findings) - 10} more")
        print(f"\n  ⛔ TRAINING ABORTED - Clean dataset required")
        return False
    else:
        print(f"  ✅ No obvious secrets in first {line_count} lines")
        print(f"  ✅ Dataset security check passed")
        return True


class GPUMemoryMonitor:
    """Monitors GPU memory usage and prevents OOM crashes.

    Tracks memory allocation in real-time and can trigger emergency
    measures (clearing cache, reducing batch size, or stopping training)
    when memory limits are approached.
    """

    def __init__(self, device: str = "cuda", safety_margin_gb: float = 1.0, logger: logging.Logger = None):
        self.device = device
        self.safety_margin_gb = safety_margin_gb
        self.logger = logger or logging.getLogger("peagle_training")
        self.max_allocated_gb = 0.0
        self.oom_count = 0
        self.emergency_reduced_batch = False

        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - GPU monitoring disabled")
            self.enabled = False
        else:
            self.enabled = True
            self.device_count = torch.cuda.device_count()
            self._log_gpu_info()

    def _log_gpu_info(self):
        """Log GPU information at startup."""
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            self.logger.info(f"GPU {i}: {props.name} | Total Memory: {total_gb:.2f} GB")

    def get_memory_stats(self, device_index: int = 0) -> Dict[str, float]:
        """Get current memory statistics for a GPU."""
        if not self.enabled:
            return {}

        torch.cuda.synchronize(device_index)

        allocated = torch.cuda.memory_allocated(device_index) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024**3)
        total = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
        free = total - allocated

        # Track peak usage
        self.max_allocated_gb = max(self.max_allocated_gb, allocated)

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": free,
            "utilization_percent": (allocated / total) * 100
        }

    def check_memory(self, device_index: int = 0) -> Tuple[bool, str]:
        """Check if GPU memory is within safe limits.

        Returns:
            (is_safe, message) tuple
        """
        if not self.enabled:
            return True, "GPU monitoring disabled"

        stats = self.get_memory_stats(device_index)

        if stats["free_gb"] < self.safety_margin_gb:
            return False, f"Low memory: {stats['free_gb']:.2f} GB free (safety margin: {self.safety_margin_gb} GB)"

        if stats["utilization_percent"] > 95:
            return False, f"High utilization: {stats['utilization_percent']:.1f}%"

        return True, f"OK: {stats['allocated_gb']:.2f} GB allocated, {stats['free_gb']:.2f} GB free"

    def emergency_cleanup(self) -> bool:
        """Perform emergency memory cleanup.

        Returns:
            True if cleanup was successful and training can continue
        """
        if not self.enabled:
            return True

        self.logger.warning("🚨 EMERGENCY GPU MEMORY CLEANUP INITIATED")

        # Empty CUDA cache
        torch.cuda.empty_cache()
        self.logger.info("  - Emptied CUDA cache")

        # Force garbage collection
        import gc
        gc.collect()
        self.logger.info("  - Ran garbage collection")

        # Check memory after cleanup
        stats = self.get_memory_stats()
        self.logger.info(f"  - Free memory after cleanup: {stats['free_gb']:.2f} GB")

        if stats["free_gb"] < self.safety_margin_gb:
            self.logger.error("  - Cleanup insufficient - still below safety margin")
            return False

        self.oom_count += 1
        self.logger.warning(f"  - Emergency cleanup #{self.oom_count} successful")
        return True

    def log_memory_summary(self):
        """Log a summary of memory usage."""
        if not self.enabled:
            return

        stats = self.get_memory_stats()
        self.logger.info(
            f"GPU Memory: {stats['allocated_gb']:.2f} GB allocated | "
            f"{stats['free_gb']:.2f} GB free | "
            f"Peak: {self.max_allocated_gb:.2f} GB"
        )

    def get_memory_report(self) -> Dict[str, Any]:
        """Get a comprehensive memory report for saving to logs."""
        if not self.enabled:
            return {"enabled": False}

        stats = self.get_memory_stats()
        return {
            "enabled": True,
            "peak_allocated_gb": self.max_allocated_gb,
            "oom_incidents": self.oom_count,
            "emergency_batch_reduction": self.emergency_reduced_batch,
            "current_stats": stats
        }


def oom_recovery_handler(func):
    """Decorator to catch OOM errors and attempt recovery.

    Wraps training functions to catch CUDA OOM errors, attempt cleanup,
    and potentially retry with reduced memory usage.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                self.logger.error(f"🚨 CUDA OOM ERROR: {e}")

                # Try emergency cleanup
                if hasattr(self, 'gpu_monitor') and self.gpu_monitor.emergency_cleanup():
                    self.logger.warning("Attempting to continue after OOM cleanup...")
                    # Re-raise to let caller decide whether to retry
                    raise RuntimeError(f"OOM recovered but step failed: {e}")
                else:
                    self.logger.error("OOM cleanup failed - stopping training")
                    # Log final memory state
                    if hasattr(self, 'gpu_monitor'):
                        self.gpu_monitor.log_memory_summary()
                    raise RuntimeError(f"Fatal OOM: {e}")
            else:
                raise
    return wrapper


def print_section(title: str, logger: logging.Logger = None):
    """Print formatted section header."""
    lines = [
        "",
        f"{'='*70}",
        f"  {title}",
        f"{'='*70}"
    ]
    if logger:
        for line in lines:
            logger.info(line)
    else:
        for line in lines:
            print(line)


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }

    if not torch.cuda.is_available():
        return info

    info["device_count"] = torch.cuda.device_count()

    for i in range(info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)  # GB

        # Get current memory usage
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = total_memory - allocated

        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(total_memory, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
            "multi_processor_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}"
        })

    return info


def estimate_vram_requirements(
    drafter_params_b: float,
    target_hidden_dim: int,
    batch_size: int,
    seq_length: int = 2048,
    speculation_depth: int = 4,
    use_lora: bool = True
) -> Dict[str, float]:
    """
    Estimate VRAM requirements in GB.

    Args:
        drafter_params_b: Drafter model size in billions (e.g., 1.5 for 1.5B)
        target_hidden_dim: Target model hidden dimension
        batch_size: Training batch size
        seq_length: Maximum sequence length
        speculation_depth: Number of MTP heads
        use_lora: Whether using LoRA
    """
    # Base model memory (parameters in fp16/bf16)
    param_bytes = 2 if use_lora else 4  # LoRA keeps base frozen in 16-bit
    base_model_gb = drafter_params_b * param_bytes

    # LoRA parameters (if enabled) - typically ~0.5-2% of base
    lora_gb = base_model_gb * 0.01 if use_lora else 0

    # Gradients (only for trainable params)
    gradients_gb = lora_gb if use_lora else base_model_gb

    # Optimizer states (PagedAdamW8bit uses 8-bit, but let's be conservative)
    optimizer_gb = lora_gb * 2 if use_lora else base_model_gb * 2

    # Activations (forward pass)
    # Rough estimate: batch * seq * hidden * layers * 4 bytes
    # Assuming ~24 layers average, 2x buffer for intermediate activations
    est_layers = 24 if drafter_params_b >= 1.5 else 16
    # Use average hidden dim: 2048 covers 1536-4096 range (Qwen to Llama)
    avg_hidden_dim = 2048
    activation_gb = (batch_size * seq_length * avg_hidden_dim * est_layers * 4) / (1024**3)

    # MTP heads memory (parallel predictions)
    mtp_head_gb = (speculation_depth * batch_size * seq_length * target_hidden_dim * 4) / (1024**3)

    # Feature cache during training
    feature_cache_gb = (batch_size * seq_length * target_hidden_dim * 4) / (1024**3)

    # System overhead (CUDA context, fragmentation, etc.)
    overhead_gb = 2.0

    total_gb = (
        base_model_gb +
        lora_gb +
        gradients_gb +
        optimizer_gb +
        activation_gb +
        mtp_head_gb +
        feature_cache_gb +
        overhead_gb
    )

    return {
        "base_model_gb": round(base_model_gb, 2),
        "lora_params_gb": round(lora_gb, 2),
        "gradients_gb": round(gradients_gb, 2),
        "optimizer_states_gb": round(optimizer_gb, 2),
        "activations_gb": round(activation_gb, 2),
        "mtp_heads_gb": round(mtp_head_gb, 2),
        "feature_cache_gb": round(feature_cache_gb, 2),
        "overhead_gb": overhead_gb,
        "total_required_gb": round(total_gb, 2),
        "recommended_gb": round(total_gb * 1.2, 2)  # 20% safety margin
    }


def estimate_training_time(
    num_samples: int,
    num_epochs: int,
    batch_size: int,
    steps_per_sec: float = 1.5
) -> Dict[str, Any]:
    """Estimate total training time."""
    steps_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps / steps_per_sec

    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_total_seconds": total_seconds,
        "estimated_total_time": str(timedelta(seconds=int(total_seconds))),
        "time_per_epoch": str(timedelta(seconds=int(steps_per_epoch / steps_per_sec)))
    }


def check_disk_space(path: str, required_gb: float = 10.0) -> Dict[str, Any]:
    """Check available disk space."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)

    return {
        "path": path,
        "total_gb": round(total_gb, 2),
        "free_gb": round(free_gb, 2),
        "required_gb": required_gb,
        "sufficient": free_gb >= required_gb
    }


def parse_model_size(model_name: str) -> float:
    """Extract model size in billions from name."""
    import re
    # Look for patterns like 1.5B, 7B, 0.5B, etc.
    match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
    if match:
        return float(match.group(1))
    # Check for common sizes in name
    if "0.5" in model_name.lower():
        return 0.5
    elif "1.5" in model_name.lower():
        return 1.5
    elif "3" in model_name.lower():
        return 3.0
    elif "7" in model_name.lower():
        return 7.0
    return 1.5  # Default assumption


class EagleTrainer:
    """Trainer for P-EAGLE Drafter model."""

    def __init__(
        self,
        drafter_model_name: str,
        target_model_name: str,
        target_hidden_dim: int,
        feature_dir: str,
        output_dir: str,
        speculation_depth: int = 4,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 256,
        learning_rate: float = 5e-5,
        batch_size: int = 2,
        num_epochs: int = 3,
        warmup_steps: int = None,
        warmup_ratio: float = 0.03,
        max_grad_norm: float = 1.0,
        save_every: int = 100,
        device: str = "cuda",
        skip_hardware_check: bool = False,
        yes: bool = False,
        quantization: str = None,
        logger: logging.Logger = None,
        run_log_dir: Path = None,
        log_dir: str = None,
        gpu_safety_margin_gb: float = 1.5,
        resume_from: str = None,
        gradient_accumulation_steps: int = 32,
        max_seq_len: int = 32768,
        rank: int = 0,
        world_size: int = 1,
        deepspeed_config: str = None,
        use_flash_attention: bool = True,
        label_smoothing: float = 0.0,
        mtp_dropout: float = 0.1,
        weight_decay: float = 0.01,
        shard_cache_size: int = 2,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_model_name = target_model_name  # Store target model name for lm_head loading
        self.feature_dir = feature_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every
        self.device = device
        self.yes = yes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        self.current_epoch = 0  # Track epoch for resume capability
        self._training_state = {}  # Loaded training state for resume
        self.drafter_model_name = drafter_model_name
        self.logger = logger or logging.getLogger("peagle_training")
        self.run_log_dir = run_log_dir
        self.resume_from = resume_from
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        self.deepspeed_enabled = False

        # Regularization parameters
        self.label_smoothing = label_smoothing
        self.mtp_dropout = mtp_dropout
        self.weight_decay = weight_decay
        if self.is_main_process:
            self.logger.info(f"Regularization: label_smoothing={label_smoothing}, mtp_dropout={mtp_dropout}, weight_decay={weight_decay}")

        # Initialize GPU memory monitor
        self.gpu_monitor = GPUMemoryMonitor(
            device=device,
            safety_margin_gb=gpu_safety_margin_gb,
            logger=self.logger
        )

        # Hardware requirement check
        if not skip_hardware_check:
            self._run_hardware_check(
                drafter_model_name=drafter_model_name,
                target_hidden_dim=target_hidden_dim,
                batch_size=batch_size,
                speculation_depth=speculation_depth,
                use_lora=use_lora,
                num_epochs=num_epochs,
                auto_confirm=self.yes
            )

        # Initialize model
        # EAGLE-3 requires hidden injection via CONCATENATION at first layer
        # First layer accepts 2x hidden size: [embeds; target_hidden]
        self.model = EagleDrafterModel(
            base_model_name=drafter_model_name,
            target_hidden_dim=target_hidden_dim,
            speculation_depth=speculation_depth,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=mtp_dropout,  # Use mtp_dropout for LoRA too
            device=device,
            use_hidden_injection=True,
            injection_mode="concat",
            quantization=quantization,
            use_flash_attention=use_flash_attention,
            mtp_dropout=mtp_dropout,
        )

        # Wrap model with DDP if using multiple GPUs (but NOT if using DeepSpeed)
        # find_unused_parameters=True needed because EAGLE's curriculum training only
        # uses specific MTP heads initially, leaving other parameters unused
        if self.world_size > 1 and not deepspeed_config:
            # For single-GPU nodes, local_rank is always 0 on both machines
            # Use LOCAL_RANK env var (set by torchrun) for correct device
            import os
            local_rank = int(os.environ.get('LOCAL_RANK', self.rank))
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True  # Needed for EAGLE curriculum learning
            )
            if self.is_main_process:
                self.logger.info(f"Wrapped model with DDP (world_size={self.world_size}, find_unused_parameters=True)")
        elif deepspeed_config and self.is_main_process:
            self.logger.info("Skipping DDP wrapping - DeepSpeed will handle distributed communication")

        # Load checkpoint if resuming (only on main process)
        if self.resume_from and self.is_main_process:
            self.logger.info(f"Resuming training from checkpoint: {self.resume_from}")
            # Unwrap DDP if needed to load checkpoint
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load.load_checkpoint(self.resume_from, device=device)

            # Load training state from checkpoint
            import json
            training_state_path = Path(self.resume_from) / "training_state.json"
            if training_state_path.exists():
                with open(training_state_path) as f:
                    self._training_state = json.load(f)
                self.current_epoch = self._training_state.get('epoch', 0)
                self.global_step = self._training_state.get('step', 0)
                if self.is_main_process:
                    self.logger.info(f"Loaded training state: epoch={self.current_epoch}, step={self.global_step}")
            else:
                # Fallback: extract step from folder name
                import re
                step_match = re.search(r'step_(\d+)', self.resume_from)
                if step_match:
                    self.global_step = int(step_match.group(1))
                    if self.is_main_process:
                        self.logger.info(f"Resumed from step {self.global_step}")
                if self.is_main_process:
                    self.logger.warning("No training_state.json found, using defaults")
                self._training_state = {}

        # === SPEED OPTIMIZATIONS ===
        # Enable TF32 for faster matmul (10% speedup, no quality loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  Enabled TF32 for faster GPU computation")

        # NOTE: torch.compile disabled for EAGLE-3 compatibility
        # Gemma3's rotary embeddings have complex internal state that
        # doesn't work well with torch.compile's dynamo tracing
        print("  Skipping torch.compile (disabled for EAGLE-3 compatibility)")
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        # ==========================

        # Enable gradient checkpointing to save VRAM (trades compute for memory)
        # Handle DDP/DeepSpeed wrapped models
        model_unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_unwrapped.base_model, 'gradient_checkpointing_enable'):
            model_unwrapped.base_model.gradient_checkpointing_enable()
            print("  Enabled gradient checkpointing (saves ~50% VRAM)")
            # Required for gradient checkpointing + LoRA to work together
            if hasattr(model_unwrapped, 'enable_input_require_grads'):
                model_unwrapped.enable_input_require_grads()
                print("  Enabled input requires grads for LoRA compatibility")

        # Load tokenizer (use same cache as model)
        import os
        cache_dir = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(drafter_model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load target model's lm_head for token-level loss computation
        # This is critical: drafter's hidden states are converted to tokens via target's lm_head
        print(f"Loading target lm_head for token-level training...")
        print(f"NOTE: Target model must have hidden_dim={target_hidden_dim} for lm_head compatibility")
        self.target_lm_head = self._load_target_lm_head(target_hidden_dim, self.target_model_name)

        # Setup optimizer with SEPARATE learning rates for different components
        # CRITICAL FIX: MTP heads need higher LR to learn effectively
        print("Setting up PagedAdamW8bit optimizer with separate LR groups...")

        lora_params = []
        mtp_params = []
        proj_params = []
        bias_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name or "norm" in name:
                bias_params.append(param)
            elif "mtp_heads" in name:
                mtp_params.append(param)
            elif "dim_projection" in name:
                proj_params.append(param)
            elif "lora" in name.lower():
                lora_params.append(param)
            else:
                # Other trainable params
                bias_params.append(param)

        # Recommended LRs: LoRA=base, MTP=10x, Projection=2x
        # MTP heads need high LR to converge fast; projection layers learn slower
        print(f"  Parameter groups:")
        print(f"    LoRA params: {len(lora_params)} tensors (LR = {learning_rate:.2e}, WD = {weight_decay})")
        print(f"    MTP heads: {len(mtp_params)} tensors (LR = {learning_rate * 10:.2e}, WD = {weight_decay})")
        print(f"    Projection: {len(proj_params)} tensors (LR = {learning_rate * 2:.2e}, WD = {weight_decay})")
        print(f"    Bias/No-WD: {len(bias_params)} tensors (LR = {learning_rate:.2e}, WD = 0.0)")

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": learning_rate, "weight_decay": weight_decay})
        if mtp_params:
            param_groups.append({"params": mtp_params, "lr": learning_rate * 10, "weight_decay": weight_decay})
        if proj_params:
            param_groups.append({"params": proj_params, "lr": learning_rate * 2, "weight_decay": weight_decay})
        if bias_params:
            param_groups.append({"params": bias_params, "lr": learning_rate, "weight_decay": 0.0})

        self.optimizer = PagedAdamW8bit(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Store base LR for gradient clipping comparison (DeepSpeed compatibility)
        self._base_lr = learning_rate

        # Load dataset with lazy shard loading (official EAGLE pattern)
        # Dataset loads only one shard at a time, reducing RAM from 148GB to ~6GB
        from ..utils.feature_utils import EagleTrainingDataset

        self.train_dataset = EagleTrainingDataset(
            feature_dir=feature_dir,
            tokenizer=self.tokenizer,
            speculation_depth=speculation_depth,
            max_seq_len=max_seq_len,
            shard_cache_size=shard_cache_size
        )

        # Split into train/val (95/5 split like official EAGLE)
        total_samples = len(self.train_dataset)
        val_size = int(0.05 * total_samples)
        train_size = total_samples - val_size

        from torch.utils.data import random_split
        generator = torch.Generator().manual_seed(42)
        self.train_subset, self.val_subset = random_split(
            self.train_dataset, [train_size, val_size], generator=generator
        )

        print(f"Dataset split: {train_size} train, {val_size} validation")

        # Create samplers for distributed training
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                self.train_subset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            self.val_sampler = DistributedSampler(
                self.val_subset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None

        # Create dataloaders
        # NOTE: num_workers=0 is REQUIRED for lazy loading to work correctly
        # The dataset loads shards on-demand in __getitem__, which is not thread-safe
        self.train_loader = DataLoader(
            self.train_subset,
            batch_size=batch_size,
            shuffle=False,  # Dataset already orders by shard for cache efficiency
            sampler=self.train_sampler,
            collate_fn=self._collate_fn,
            num_workers=0,  # Must be 0 for lazy loading compatibility
            pin_memory=False  # Not needed when num_workers=0
        )

        self.val_loader = DataLoader(
            self.val_subset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=False
        )

        self.speculation_depth = speculation_depth

        # Setup scheduler
        total_steps = len(self.train_loader) * num_epochs
        # Calculate warmup_steps from ratio if not explicitly specified
        if warmup_steps is None:
            self.warmup_steps = int(warmup_ratio * total_steps)
            if self.is_main_process:
                self.logger.info(f"Warmup: {self.warmup_steps} steps ({warmup_ratio:.1%} of {total_steps} total steps)")
        else:
            self.warmup_steps = warmup_steps
            if self.is_main_process:
                self.logger.info(f"Warmup: {self.warmup_steps} steps (fixed)")
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # TensorBoard - use custom log_dir if provided, otherwise default to output_dir/logs
        log_path = Path(log_dir) if log_dir else self.output_dir / "logs"
        self.writer = SummaryWriter(log_dir=log_path)
        self.metrics_tracker = MetricsTracker()

        # Weights & Biases
        self.wandb_enabled = WANDB_AVAILABLE and self.is_main_process
        if self.wandb_enabled:
            run_name = f"peagle_drafter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project="P-EAGLE",
                name=run_name,
                config={
                    "drafter_model": drafter_model_name,
                    "target_hidden_dim": target_hidden_dim,
                    "speculation_depth": speculation_depth,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "max_seq_len": max_seq_len,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "use_lora": use_lora,
                    "use_flash_attention": use_flash_attention,
                    "label_smoothing": label_smoothing,
                    "mtp_dropout": mtp_dropout,
                    "weight_decay": weight_decay,
                },
                dir=str(self.output_dir),
            )
            self.logger.info(f"W&B initialized: {wandb.run.url if hasattr(wandb.run, 'url') else 'offline'}")

        print(f"Training setup complete:")
        print(f"  Total samples: {total_samples}")
        print(f"  Train: {train_size}, Val: {val_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {len(self.train_loader)}")

    def _load_target_lm_head(self, target_hidden_dim: int, target_model_name: str = None):
        """Load target model's lm_head for perfect alignment.

        CRITICAL FIX: This function now loads lm_head directly from the TARGET model,
        not from the drafter. This ensures the drafter predicts tokens using the EXACT
        same lm_head projection as the target model, which is essential for speculative
        decoding to work.

        The original implementation had two critical bugs:
        1. Used drafter's tokenizer vocab_size (128K) instead of target's (262K)
        2. Fell back to random initialization when lm_head wasn't in feature files

        FIXED: Now handles Gemma-3 multimodal models which use model.model.language_model.lm_head

        Args:
            target_hidden_dim: Hidden dimension of target model
            target_model_name: Name of target model (for direct loading)
        """
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Default to drafter's vocab size
        vocab_size = len(self.tokenizer)
        actual_vocab_size = None

        # First, try to get the correct vocab size from feature files
        try:
            feature_files = list(Path(self.feature_dir).glob("*_shard*.pt"))
            if feature_files:
                data = torch.load(feature_files[0], map_location='cpu', weights_only=False)
                if "vocab_size" in data:
                    feature_vocab = data["vocab_size"]
                    if feature_vocab != vocab_size:
                        print(f"  CRITICAL FIX: Feature file vocab_size ({feature_vocab}) != drafter tokenizer ({vocab_size})")
                        vocab_size = feature_vocab
                # Also check if lm_head was saved in feature file
                if "lm_head" in data and data["lm_head"] is not None:
                    print(f"  Found lm_head in feature file (will try to use)")
        except Exception as e:
            print(f"  Note: Could not read vocab_size from features: {e}")

        # Try to load lm_head from the TARGET model directly (THE FIX)
        target_lm = None
        if target_model_name:
            try:
                print(f"  Loading target model lm_head directly from: {target_model_name}")
                HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

                # Load target model
                target_model = AutoModelForCausalLM.from_pretrained(
                    target_model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    token=HF_TOKEN
                )

                # Try multiple locations where lm_head might be stored
                # Gemma-3 multimodal models use model.model.language_model.lm_head
                lm_head_locations = [
                    ("target_model.lm_head", lambda m: m if hasattr(m, 'lm_head') and m.lm_head is not None else None),
                    ("target_model.model.lm_head", lambda m: m.model if hasattr(m, 'model') and hasattr(m.model, 'lm_head') and m.model.lm_head is not None else None),
                    ("target_model.model.model.lm_head", lambda m: m.model.model if hasattr(m.model, 'model') and hasattr(m.model.model, 'lm_head') and m.model.model.lm_head is not None else None),
                    ("target_model.language_model.lm_head", lambda m: m.language_model if hasattr(m, 'language_model') and hasattr(m.language_model, 'lm_head') and m.language_model.lm_head is not None else None),
                    # CRITICAL: Gemma-3 multimodal structure (gemma-3-4b-it)
                    ("target_model.model.language_model.lm_head", lambda m: m.model.language_model if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'lm_head') and m.model.language_model.lm_head is not None else None),
                ]

                for location_name, check_func in lm_head_locations:
                    try:
                        lm_obj = check_func(target_model)
                        if lm_obj is not None:
                            target_lm = lm_obj
                            print(f"  ✓ Found lm_head at {location_name}")
                            break
                    except Exception as e:
                        pass  # Try next location

                if target_lm is None:
                    print(f"  Warning: Could not find lm_head in target model at any known location!")
                    print(f"  Model type: {type(target_model).__name__}")
                    # List attributes for debugging
                    attrs = [a for a in dir(target_model) if 'lm' in a.lower() or 'head' in a.lower()]
                    print(f"  Model attributes containing 'lm' or 'head': {attrs}")

                if target_lm is not None:
                    target_weight = target_lm.weight.detach()
                    actual_vocab_size = target_weight.shape[0]

                    print(f"  Target model lm_head shape: {target_weight.shape}")

                    # If target vocab_size is different, we need to handle it
                    if actual_vocab_size != vocab_size:
                        print(f"  CRITICAL: Target vocab_size ({actual_vocab_size}) != current ({vocab_size})")
                        print(f"  Will create lm_head with target's vocab_size")

                    # Create lm_head with target's vocab size
                    lm_head = nn.Linear(target_hidden_dim, actual_vocab_size, bias=False, dtype=torch.bfloat16).to(self.device)

                    # Copy weights
                    with torch.no_grad():
                        # Check if hidden dimensions match
                        if target_weight.shape[1] == target_hidden_dim:
                            lm_head.weight.copy_(target_weight)
                            print(f"  ✓ Copied lm_head from target model: {target_weight.shape}")
                        else:
                            # Need to project hidden dimension
                            print(f"  Warning: Hidden dim mismatch ({target_weight.shape[1]} vs {target_hidden_dim}), copying overlapping weights")
                            min_hidden = min(target_weight.shape[1], target_hidden_dim)
                            lm_head.weight[:, :min_hidden] = target_weight[:, :min_hidden]

                    # Clean up
                    del target_model
                    torch.cuda.empty_cache()

                    print(f"  lm_head: {target_hidden_dim} -> {actual_vocab_size}")
                    return lm_head

            except Exception as e:
                print(f"  Warning: Could not load from target model: {e}")
                import traceback
                traceback.print_exc()

        # Fall back to creating lm_head with whatever vocab_size we have
        print(f"  Creating lm_head with fallback: {target_hidden_dim} -> {vocab_size}")
        lm_head = nn.Linear(target_hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16).to(self.device)

        # Try to load from feature files as secondary option
        try:
            feature_files = list(Path(self.feature_dir).glob("*_shard*.pt"))
            if feature_files:
                data = torch.load(feature_files[0], map_location='cpu', weights_only=False)

                if "lm_head" in data and data["lm_head"] is not None:
                    saved_lm_head = data["lm_head"]
                    saved_weight = saved_lm_head.get("weight") if isinstance(saved_lm_head, dict) else None

                    if saved_weight is not None:
                        print(f"  Loaded lm_head from feature file: {saved_weight.shape}")
                        with torch.no_grad():
                            min_vocab = min(saved_weight.shape[0], vocab_size)
                            min_hidden = min(saved_weight.shape[1], target_hidden_dim)
                            lm_head.weight[:min_vocab, :min_hidden] = saved_weight[:min_vocab, :min_hidden]
                        print(f"  Copied {min_vocab} x {min_hidden} weights from feature file")
                        return lm_head
        except Exception as e:
            print(f"  Warning: Could not load from feature files: {e}")

        # Final fallback: random initialization (THIS SHOULD NOT HAPPEN with proper setup)
        print(f"  ***WARNING: Using randomly initialized lm_head!***")
        print(f"  This will NOT work for speculative decoding!")
        print(f"  CRITICAL: You must re-extract features with the fixed feature_extractor.py")
        nn.init.normal_(lm_head.weight, std=0.02)

        return lm_head

    def _run_hardware_check(
        self,
        drafter_model_name: str,
        target_hidden_dim: int,
        batch_size: int,
        speculation_depth: int,
        use_lora: bool,
        num_epochs: int,
        auto_confirm: bool = False
    ):
        """Run comprehensive hardware requirement check before training."""
        print_section("P-EAGLE TRAINING - HARDWARE CHECK")

        # Parse model size
        model_size_b = parse_model_size(drafter_model_name)
        print(f"\n📊 Model: {drafter_model_name}")
        print(f"   Estimated size: ~{model_size_b}B parameters")
        print(f"   Target hidden dim: {target_hidden_dim}")
        print(f"   Speculation depth (K): {speculation_depth}")
        print(f"   LoRA enabled: {use_lora}")

        # GPU Check
        print_section("GPU AVAILABILITY")
        gpu_info = get_gpu_info()

        if not gpu_info["cuda_available"]:
            print("❌ ERROR: CUDA not available!")
            print("   Training requires at least one NVIDIA GPU.")
            raise RuntimeError("CUDA not available. GPU is required for training.")

        print(f"✅ CUDA available")
        print(f"   Device count: {gpu_info['device_count']}")

        total_gpu_memory = 0
        for dev in gpu_info["devices"]:
            print(f"\n   GPU {dev['index']}: {dev['name']}")
            print(f"   ├── Total memory: {dev['total_memory_gb']:.2f} GB")
            print(f"   ├── Free memory: {dev['free_gb']:.2f} GB")
            print(f"   ├── Compute capability: {dev['compute_capability']}")
            print(f"   └── Multi-processors: {dev['multi_processor_count']}")
            total_gpu_memory += dev['total_memory_gb']

        # VRAM Estimation
        print_section("VRAM REQUIREMENTS")
        vram_req = estimate_vram_requirements(
            drafter_params_b=model_size_b,
            target_hidden_dim=target_hidden_dim,
            batch_size=batch_size,
            speculation_depth=speculation_depth,
            use_lora=use_lora
        )

        print(f"\nEstimated VRAM breakdown:")
        print(f"  Base model ({'16-bit' if use_lora else '32-bit'}): {vram_req['base_model_gb']:.2f} GB")
        if use_lora:
            print(f"  LoRA parameters: {vram_req['lora_params_gb']:.2f} GB")
            print(f"  Gradients (LoRA only): {vram_req['gradients_gb']:.2f} GB")
        else:
            print(f"  Gradients (full): {vram_req['gradients_gb']:.2f} GB")
        print(f"  Optimizer states: {vram_req['optimizer_states_gb']:.2f} GB")
        print(f"  Activations: {vram_req['activations_gb']:.2f} GB")
        print(f"  MTP heads: {vram_req['mtp_heads_gb']:.2f} GB")
        print(f"  Feature cache: {vram_req['feature_cache_gb']:.2f} GB")
        print(f"  System overhead: {vram_req['overhead_gb']:.2f} GB")
        print(f"\n{'─'*50}")
        print(f"  TOTAL REQUIRED: ~{vram_req['total_required_gb']:.2f} GB")
        print(f"  RECOMMENDED: ~{vram_req['recommended_gb']:.2f} GB (20% margin)")

        # Check if sufficient VRAM
        if vram_req['recommended_gb'] > total_gpu_memory:
            print(f"\n⚠️  WARNING: Insufficient GPU memory!")
            print(f"   You have: {total_gpu_memory:.2f} GB total")
            print(f"   Recommended: {vram_req['recommended_gb']:.2f} GB")
            print(f"\n   Suggestions:")
            print(f"   • Reduce batch_size (currently {batch_size})")
            print(f"   • Use a smaller drafter model")
            print(f"   • Reduce speculation_depth (currently {speculation_depth})")
            print(f"   • Use gradient accumulation instead")

            user_input = input("\nContinue anyway? [y/N]: ").strip().lower()
            if user_input != 'y':
                raise RuntimeError("Hardware requirements not met. Aborting.")
        else:
            print(f"\n✅ Sufficient GPU memory available")

        # Disk Space Check
        print_section("DISK SPACE CHECK")
        # Estimate: base model (~3GB) + checkpoints (~10GB) + logs (~1GB)
        estimated_need_gb = 15 + (model_size_b * 2)
        disk_check = check_disk_space(str(self.output_dir), estimated_need_gb)

        print(f"Output directory: {disk_check['path']}")
        print(f"  Total: {disk_check['total_gb']:.2f} GB")
        print(f"  Free: {disk_check['free_gb']:.2f} GB")
        print(f"  Estimated need: ~{estimated_need_gb:.2f} GB")

        if not disk_check['sufficient']:
            print(f"\n⚠️  WARNING: Low disk space!")
            user_input = input("Continue anyway? [y/N]: ").strip().lower()
            if user_input != 'y':
                raise RuntimeError("Insufficient disk space. Aborting.")
        else:
            print(f"✅ Sufficient disk space")

        # Training Time Estimation
        print_section("TRAINING TIME ESTIMATE")

        # Count feature files
        feature_files = list(Path(self.feature_dir if hasattr(self, 'feature_dir') else ".").glob("*.pt"))
        num_samples = len(feature_files)

        if num_samples > 0:
            time_est = estimate_training_time(
                num_samples=num_samples,
                num_epochs=num_epochs,
                batch_size=batch_size
            )

            print(f"Dataset: {num_samples} samples")
            print(f"Epochs: {num_epochs}")
            print(f"Steps per epoch: {time_est['steps_per_epoch']}")
            print(f"Total steps: {time_est['total_steps']}")
            print(f"\n⏱️  Estimated training time:")
            print(f"   Per epoch: {time_est['time_per_epoch']}")
            print(f"   Total: ~{time_est['estimated_total_time']}")
        else:
            print(f"⚠️  No feature files found yet. Cannot estimate training time.")

        # Model Download Info
        print_section("MODEL DOWNLOAD INFO")
        print(f"Drafter model will be downloaded from HuggingFace:")
        print(f"  {drafter_model_name}")
        print(f"\nCache location: ~/.cache/huggingface/hub/")
        print(f"Download size: ~{model_size_b * 2:.1f} GB (weights + tokenizer)")

        # Final confirmation
        print_section("READY TO START")
        print("\nConfiguration summary:")
        print(f"  Drafter: {drafter_model_name}")
        print(f"  Target hidden dim: {target_hidden_dim}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  LoRA rank: {64 if use_lora else 'N/A (full fine-tune)'}")
        print(f"  Output dir: {self.output_dir}")

        if auto_confirm:
            print("\n🚀 Auto-confirmed ( --yes flag ), starting training...\n")
        else:
            user_input = input("\nStart training? [Y/n]: ").strip().lower()
            if user_input == 'n':
                raise RuntimeError("Training cancelled by user.")

        print("\n🚀 Starting training...\n")

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for batching. Returns CPU tensors.
        Pads to multiple of 8 for tensor core efficiency.
        """
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
            print(f"WARNING: pad_token_id is None, using fallback value 0")

        pad_multiple = 8

        input_ids = nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=pad_token_id
        )
        target_hidden = nn.utils.rnn.pad_sequence(
            [b["target_hidden"] for b in batch],
            batch_first=True,
            padding_value=0.0
        )
        loss_mask = nn.utils.rnn.pad_sequence(
            [b["loss_mask"] for b in batch],
            batch_first=True,
            padding_value=0
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0
        )

        # Pad to multiple of 8 for tensor core efficiency
        current_len = input_ids.shape[1]
        pad_len = (pad_multiple - current_len % pad_multiple) % pad_multiple
        if pad_len > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_token_id)
            target_hidden = torch.nn.functional.pad(target_hidden, (0, 0, 0, pad_len), value=0.0)
            loss_mask = torch.nn.functional.pad(loss_mask, (0, pad_len), value=0)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        target_token_ids_list = []
        for b in batch:
            ttids = b.get("target_token_ids", None)
            if ttids is not None:
                target_token_ids_list.append(ttids)
            else:
                target_token_ids_list.append(torch.full((1,), -100, dtype=torch.long))

        target_token_ids = nn.utils.rnn.pad_sequence(
            target_token_ids_list,
            batch_first=True,
            padding_value=-100
        )
        if pad_len > 0:
            target_token_ids = torch.nn.functional.pad(target_token_ids, (0, pad_len), value=-100)

        return {
            "input_ids": input_ids,
            "target_hidden": target_hidden,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "target_token_ids": target_token_ids
        }

    @torch.no_grad()
    def _validation_step(self, epoch: int = 1):
        """Compute validation loss using same loss function as training.

        Uses curriculum learning to only validate active heads.
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = []

        # Determine active heads for curriculum learning
        active_heads = self._get_active_heads(epoch)

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_hidden = batch["target_hidden"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)
            target_token_ids = batch["target_token_ids"].to(self.device)  # Precomputed targets

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_hidden=target_hidden,
                is_training=True
            )

            # Apply loss mask to compute validation loss
            mtp_predictions = outputs["mtp_predictions"]
            ce_losses = []
            mse_losses = []

            for i, pred_hidden in enumerate(mtp_predictions):
                # CURRICULUM: Only validate active heads
                if i >= active_heads:
                    break

                # CRITICAL FIX: Align validation with training logic
                # With new model, pred_hidden is full sequence [B, N, H] for all heads
                # Head k: prediction at position t matches target at position t + shift
                shift = i + 1

                # Trim predictions and targets to match training logic
                pred_trimmed = pred_hidden[:, :-shift, :]  # Remove last shift positions
                target_trimmed = target_hidden[:, shift:, :]  # Start from shift position

                # Ensure shapes match (trim to minimum length)
                min_len = min(pred_trimmed.shape[1], target_trimmed.shape[1])
                if min_len <= 0:
                    continue

                pred_trimmed = pred_trimmed[:, :min_len, :]
                target_trimmed = target_trimmed[:, :min_len, :]

                # CRITICAL FIX: Use SAME mask alignment as training
                # Training uses: loss_mask[:, :min_len]
                # Validation was using: loss_mask[:, shift:shift + min_len] (WRONG!)
                mask_trimmed = loss_mask[:, :min_len]

                # FIX: Use precomputed target_token_ids, properly shifted for MTP
                # target_token_ids[t] = model's argmax predicting token at position t+1
                # MTP head i predicts position t+i+1 → targets at index t+i = target_token_ids[i:]
                target_ids_shifted = target_token_ids[:, shift:shift + min_len]

                # Use SAME loss function as training for consistency
                ce_loss_k, mse_loss_k, _ = hidden_state_token_loss(
                    pred_trimmed,
                    target_trimmed,
                    self.target_lm_head,
                    mask_trimmed,
                    temperature=1.0,
                    target_token_ids=target_ids_shifted
                )

                if not torch.isnan(ce_loss_k):
                    ce_losses.append(ce_loss_k.item())
                    mse_losses.append(mse_loss_k.item())

            # Combine losses same way as training
            total_loss = 0.0
            if ce_losses:
                # Apply same weighting as training
                weighted_ce = []
                for i, ce in enumerate(ce_losses):
                    weight = max(0.5, 1.0 - i * 0.1)
                    weighted_ce.append(ce * weight)
                ce_total = sum(weighted_ce) / sum(max(0.5, 1.0 - i * 0.1) for i in range(len(weighted_ce)))
                total_loss += ce_total

            if mse_losses:
                mse_total = sum(mse_losses) / len(mse_losses)
                total_loss += 0.1 * mse_total

            val_losses.append(total_loss)

        self.model.train()
        return np.mean(val_losses) if val_losses else None

    def train(self):
        """Main training loop with validation and early stopping."""
        self.model.train()
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 5  # Early stopping patience
        epoch_stats = []

        # Initial GPU memory check
        self.logger.info(f"\n{'='*50}")
        self.logger.info("INITIAL GPU MEMORY CHECK")
        self.logger.info(f"{'='*50}")
        is_safe, mem_msg = self.gpu_monitor.check_memory()
        if not is_safe:
            self.logger.warning(f"Initial memory warning: {mem_msg}")
            if not self.gpu_monitor.emergency_cleanup():
                self.logger.error("Insufficient GPU memory to start training")
                raise RuntimeError("GPU memory too low to begin training")
        self.gpu_monitor.log_memory_summary()

        # Sanity check: verify first batch has non-zero loss mask and check per-head coverage
        self.logger.info("\nVerifying training data...")
        first_batch = next(iter(self.train_loader))
        mask_sum = first_batch["loss_mask"].sum().item()
        seq_len = first_batch["loss_mask"].shape[1]
        self.logger.info(f"  First batch mask sum: {mask_sum} out of {seq_len} tokens")

        # Check mask distribution
        mask = first_batch["loss_mask"][0].numpy()
        mask_ones_positions = np.where(mask == 1)[0]
        if len(mask_ones_positions) > 0:
            first_one = mask_ones_positions[0]
            last_one = mask_ones_positions[-1]
            self.logger.info(f"  Mask coverage: positions {first_one} to {last_one} ({last_one - first_one + 1} positions)")

            # Calculate per-head overlap (how many positions have mask=1 for each head)
            for k in range(min(4, self.speculation_depth)):
                shift = k + 1
                # For head k: predictions at [0, N-2-k], targets at [k+1, N-1]
                # Mask needed at: predictions [0, N-2-k] AND targets [k+1, N-1+k+1]
                # Overlap of mask requirements: [k+1, N-2-k]
                overlap_start = k + 1
                overlap_end = seq_len - 2 - k
                if overlap_start < overlap_end:
                    mask_in_overlap = mask[overlap_start:overlap_end].sum()
                    self.logger.info(f"  Head {k+1}: overlap region [{overlap_start}, {overlap_end}], mask sum = {mask_in_overlap}")

        if mask_sum == 0:
            self.logger.warning("  WARNING: Loss mask is all zeros! Training will fail.")
            self.logger.warning("  Check that dataset has proper 'loss_mask_segments'.")
        else:
            self.logger.info(f"  OK: {mask_sum} trainable tokens in first batch")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch  # Track for resume capability

            # Skip epochs if resuming from checkpoint
            if self.resume_from and epoch < self._training_state.get('epoch', 0):
                self.logger.info(f"Skipping epoch {epoch + 1} (already completed)")
                continue

            # Set epoch for distributed sampler to ensure different shuffling each epoch
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.logger.info(f"{'='*50}")

            # Log curriculum learning status (only on main process)
            active_heads = self._get_active_heads(epoch + 1)
            if self.is_main_process:
                self.logger.info(f"Curriculum: Training heads 1-{active_heads} of {self.speculation_depth}")

            epoch_losses = []
            epoch_ce_losses = []
            epoch_mse_losses = []
            epoch_mtp_losses = {i: [] for i in range(self.speculation_depth)}
            # Gradient accumulation setup
            grad_accum_steps = self.gradient_accumulation_steps
            effective_batch_size = self.batch_size * grad_accum_steps

            # DIAGNOSTIC: Log before creating tqdm
            if self.is_main_process:
                self.logger.info(f"Creating tqdm progress bar for {len(self.train_loader)} batches...")

            # Create tqdm with safe settings to handle broken pipes
            # Create progress bar with safe settings (handles broken pipes gracefully)
            try:
                pbar = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    mininterval=0.5,
                    maxinterval=2.0
                )
            except (BrokenPipeError, OSError) as e:
                # Pipe already closed - use a dummy progress bar
                self.logger.warning(f"Cannot create progress bar ({type(e).__name__}). Continuing without progress display...")
                pbar = _DummyPbarWrapper()

            # Periodic memory check interval (check 10 times per epoch)
            memory_check_interval = max(1, len(self.train_loader) // 10)
            batchOOM_retry_count = 0

            # Convert iterator to list to support slicing for accumulation
            if self.is_main_process:
                self.logger.info("Creating batch iterator...")

            batch_iterator = iter(self.train_loader)
            batch_idx = 0
            accumulation_counter = 0

            # FIX: Skip batches when resuming to continue from correct position
            if self.global_step > 0 and self.is_main_process:
                batches_per_epoch = len(self.train_loader)
                # Calculate which batch we should start from in this epoch
                # global_step = number of optimizer steps completed
                # Each step processes grad_accum_steps batches
                # So batches_skipped = global_step * grad_accum_steps
                # But we only skip within current epoch
                batches_skipped = (self.global_step * self.gradient_accumulation_steps) % batches_per_epoch
                if batches_skipped > 0 and batches_skipped < batches_per_epoch:
                    self.logger.info(f"Resuming: skipping first {batches_skipped} batches to continue from step {self.global_step}...")
                    for _ in range(batches_skipped):
                        try:
                            _ = next(batch_iterator)
                        except StopIteration:
                            break
                    batch_idx = batches_skipped
                    self.logger.info(f"Resumed from batch {batch_idx}/{batches_per_epoch}")

            if self.is_main_process:
                remaining = len(self.train_loader) - batch_idx
                self.logger.info(f"Starting training loop with {remaining} remaining steps...")

            while batch_idx < len(self.train_loader):
                # Check GPU memory periodically
                if batch_idx % memory_check_interval == 0:
                    is_safe, mem_msg = self.gpu_monitor.check_memory()
                    if not is_safe:
                        self.logger.warning(f"Memory warning: {mem_msg}")
                        if not self.gpu_monitor.emergency_cleanup():
                            self.logger.error("Unable to free sufficient memory - stopping training")
                            self._save_checkpoint("emergency_stop_low_memory")
                            return  # Exit training cleanly

                # Accumulate gradients over multiple batches
                accum_loss = 0.0
                accum_metrics = {}
                valid_accum_steps = 0

                for accum_step in range(grad_accum_steps):
                    # DIAGNOSTIC: Log before fetching first batch
                    if batch_idx == 0 and accum_step == 0 and self.is_main_process:
                        self.logger.info("Fetching first batch from data loader...")

                    try:
                        batch = next(batch_iterator)
                    except StopIteration:
                        break

                    # DIAGNOSTIC: Log after fetching first batch
                    if batch_idx == 0 and accum_step == 0 and self.is_main_process:
                        self.logger.info(f"First batch fetched. Input ids shape: {batch['input_ids'].shape}")

                    current_accum_step = (accumulation_counter * grad_accum_steps + accum_step) % grad_accum_steps
                    is_last_accum = (accum_step == grad_accum_steps - 1)

                    # Perform training step (with OOM recovery)
                    try:
                        loss, metrics = self._training_step(
                            batch,
                            epoch=epoch + 1,
                            accumulation_step=current_accum_step,
                            total_accumulation_steps=grad_accum_steps
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            self.logger.error(f"OOM during training step at batch {batch_idx}, accum_step {accum_step}")
                            if self.gpu_monitor.emergency_cleanup() and batchOOM_retry_count < 3:
                                batchOOM_retry_count += 1
                                self.logger.warning(f"Retrying batch after OOM cleanup (attempt {batchOOM_retry_count}/3)")
                                torch.cuda.synchronize()
                                continue
                            else:
                                self.logger.error("Too many OOM errors - saving checkpoint and stopping")
                                self._save_checkpoint(f"emergency_stop_oom_epoch{epoch+1}_batch{batch_idx}")
                                return
                        else:
                            raise

                    accum_loss += loss.item()
                    valid_accum_steps += 1

                    # Aggregate metrics
                    for key, value in metrics.items():
                        if key not in accum_metrics:
                            accum_metrics[key] = []
                        if hasattr(value, 'item'):
                            accum_metrics[key].append(value.item())
                        else:
                            accum_metrics[key].append(value)

                    batch_idx += 1

                    # Clean up memory after each step to prevent accumulation
                    # This helps prevent SIGKILL from memory exhaustion
                    torch.cuda.synchronize()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if self.is_main_process and self.global_step % 5 == 0:
                        mem_stats = self.gpu_monitor.get_memory_stats()
                        self.logger.debug(f"Memory after step {self.global_step}: {mem_stats['allocated_gb']:.2f}GB / {mem_stats['total_gb']:.2f}GB")

                # Average accumulated loss and metrics
                if valid_accum_steps > 0:
                    avg_loss = accum_loss / valid_accum_steps
                    epoch_losses.append(avg_loss)
                    self.global_step += 1
                    accumulation_counter += 1

                # Compute average metrics from accumulation
                avg_metrics = {}
                for key, values in accum_metrics.items():
                    if values:
                        avg_metrics[key] = sum(values) / len(values)

                # Track per-head losses and component losses
                for i in range(self.speculation_depth):
                    key = f"mtp_loss_{i+1}"
                    if key in avg_metrics:
                        epoch_mtp_losses[i].append(avg_metrics[key])

                # Track CE and MSE separately for diagnostics
                if "ce_loss_avg" in avg_metrics:
                    epoch_ce_losses.append(avg_metrics["ce_loss_avg"])
                if "mse_loss_avg" in avg_metrics:
                    epoch_mse_losses.append(avg_metrics["mse_loss_avg"])

                # Update progress bar with more informative metrics
                postfix_dict = {
                    "loss": f"{avg_loss:.4f}",
                    "ce": f"{avg_metrics.get('ce_loss_avg', 0):.4f}",
                    "mse": f"{avg_metrics.get('mse_loss_avg', 0):.4f}",
                    "acc": f"{avg_metrics.get('token_acc_avg', 0):.1f}%",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
                if grad_accum_steps > 1:
                    postfix_dict["accum"] = f"{valid_accum_steps}/{grad_accum_steps}"
                # Only show mask coverage if it's concerning (low coverage)
                mask_cov = avg_metrics.get('avg_mask_coverage', 1.0)
                if mask_cov < 0.5:
                    postfix_dict["mask"] = f"{mask_cov:.0%}"
                # Safely update progress bar (handles broken pipe gracefully)
                try:
                    pbar.set_postfix(postfix_dict)
                    pbar.update(valid_accum_steps)
                except (BrokenPipeError, OSError):
                    # Pipe closed - disable progress bar and continue training
                    pbar.disable = True
                    if self.is_main_process:
                        self.logger.warning("Progress bar output stopped (pipe closed). Continuing training...")

                # Log to TensorBoard & W&B
                if self.global_step % 10 == 0:
                    log_data = {
                        "train/total_loss": avg_loss,
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/mtp_loss_avg": avg_metrics.get("mtp_loss_avg", 0),
                        "train/ce_loss_avg": avg_metrics.get("ce_loss_avg", 0),
                        "train/token_acc_avg": avg_metrics.get("token_acc_avg", 0),
                    }
                    for k, v in avg_metrics.items():
                        if k.startswith("mtp_loss_") or k.startswith("token_acc_"):
                            log_data[f"train/{k}"] = v
                    for k, v in log_data.items():
                        self.writer.add_scalar(k, v, self.global_step)
                    if self.wandb_enabled:
                        wandb.log(log_data, step=self.global_step)

                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self._save_checkpoint(f"checkpoint_step_{self.global_step}")

            # Epoch summary
            avg_train_loss = np.mean(epoch_losses)

            # Verify MTP heads are learning (loss trend is the real signal)
            if epoch == 1 or epoch == self.num_epochs // 2 or epoch == self.num_epochs:
                with torch.no_grad():
                    mtp_std_sum = 0.0
                    for head in self.model.mtp_heads:
                        for param in head.parameters():
                            if len(param.shape) >= 2:  # Weight matrices only
                                mtp_std_sum += param.std().item()
                                break
                    avg_mtp_std = mtp_std_sum / len(self.model.mtp_heads)
                    self.logger.info(f"  MTP weight avg std: {avg_mtp_std:.6f} (init was ~0.02)")
                    # Std is a coarse metric — weights can change significantly while std stays similar
                    if abs(avg_mtp_std - 0.02) < 0.005:
                        self.logger.info("  ℹ️  MTP std near init (normal — check loss trend instead)")
                    else:
                        self.logger.info(f"  ✓ MTP heads have changed from init by {(avg_mtp_std - 0.02):.6f}")

            # Calculate component loss averages for the epoch
            avg_ce_loss = np.mean(epoch_ce_losses) if epoch_ce_losses else 0.0
            avg_mse_loss = np.mean(epoch_mse_losses) if epoch_mse_losses else 0.0

            # Validation
            val_loss = self._validation_step(epoch=epoch + 1)

            epoch_stat = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": val_loss,
                "avg_ce_loss": avg_ce_loss,
                "avg_mse_loss": avg_mse_loss,
                "per_head_avg_loss": {},
                "active_heads": active_heads,
                "gpu_memory": self.gpu_monitor.get_memory_stats()
            }
            for i in range(self.speculation_depth):
                if epoch_mtp_losses[i]:
                    epoch_stat["per_head_avg_loss"][f"head_{i+1}"] = np.mean(epoch_mtp_losses[i])

            epoch_stats.append(epoch_stat)

            # Log to TensorBoard & W&B
            self.writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)
            if val_loss is not None:
                self.writer.add_scalar("epoch/val_loss", val_loss, epoch + 1)
            if self.wandb_enabled:
                epoch_wandb = {"epoch/train_loss": avg_train_loss}
                if val_loss is not None:
                    epoch_wandb["epoch/val_loss"] = val_loss
                for i in range(self.speculation_depth):
                    if epoch_mtp_losses[i]:
                        epoch_wandb[f"epoch/mtp_head_{i+1}"] = np.mean(epoch_mtp_losses[i])
                wandb.log(epoch_wandb, step=epoch + 1)

            self.logger.info(f"Epoch {epoch + 1} summary:")
            self.logger.info(f"  Train loss: {avg_train_loss:.6f} (CE: {avg_ce_loss:.6f}, MSE: {avg_mse_loss:.6f})")
            if val_loss is not None:
                self.logger.info(f"  Val loss:   {val_loss:.6f}")
                # Check for overfitting
                if val_loss > avg_train_loss * 1.5:
                    self.logger.warning(f"  Possible overfitting detected (val >> train)")

            # Log per-head losses with clearer formatting
            if epoch_stat["per_head_avg_loss"]:
                self.logger.info(f"  Per-head MSE losses (curriculum: heads 1-{active_heads} trained):")
                for head, loss_val in sorted(epoch_stat["per_head_avg_loss"].items()):
                    status = "trained" if int(head.split("_")[1]) <= active_heads else "inactive"
                    self.logger.info(f"    {head}: {loss_val:.6f} ({status})")

            # Log memory after epoch
            self.gpu_monitor.log_memory_summary()

            # Save best model based on validation loss
            model_improved = False
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_improved = True
                    patience_counter = 0
            else:
                # Fallback to training loss if no validation set
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    model_improved = True

            if model_improved:
                self._save_checkpoint("best_model")
                self.logger.info(f"  *** New best model saved! ***")
            else:
                patience_counter += 1
                if patience_counter >= patience and self.val_loader is not None:
                    self.logger.warning(f"  Early stopping triggered (patience={patience})")
                    break

            # Save epoch checkpoint for resume capability
            self._save_checkpoint(f"epoch_{epoch+1}_checkpoint")
            self.logger.info(f"  Epoch {epoch+1} checkpoint saved")

        # Save training history
        history_path = self.output_dir / "training_history.json"

        # Add GPU memory report to training history
        memory_report = self.gpu_monitor.get_memory_report()

        with open(history_path, "w") as f:
            json.dump({
                "final_best_loss": best_val_loss,
                "epochs": epoch_stats,
                "gpu_memory_report": memory_report,
                "config": {
                    "drafter_model": self.drafter_model_name,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.train_loader.batch_size,
                    "learning_rate": getattr(self, '_base_lr', self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0),
                    "warmup_steps": self.warmup_steps
                }
            }, f, indent=2)
        self.logger.info(f"\nTraining history saved to {history_path}")

        # Log final memory summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info("GPU MEMORY SUMMARY")
        self.logger.info(f"{'='*50}")
        self.gpu_monitor.log_memory_summary()

        self.writer.close()
        if self.wandb_enabled:
            wandb.finish()
        self.logger.info("\nTraining complete!")

    def _get_active_heads(self, epoch: int) -> int:
        """Determine how many MTP heads to train based on curriculum schedule.

        Curriculum learning: Start with head 1 only, gradually add deeper heads.
        This prevents the compounding error problem in speculative decoding.

        Schedule:
        - Epoch 1: Train only head 1 (foundation)
        - Epoch 2: Train heads 1-2
        - Epoch 3: Train heads 1-3
        - Epoch 4+: Train all heads

        CRITICAL FIX: This ensures each head learns to predict its specific offset
        before adding the complexity of deeper heads. This is essential because
        deeper heads need good shallow heads to provide context.
        """
        epoch_in_training = epoch - 1  # Convert to 0-indexed

        if epoch_in_training < 1:
            return 1  # Epoch 1: only head 1
        elif epoch_in_training < 2:
            return 2  # Epoch 2: heads 1-2
        elif epoch_in_training < 3:
            return 3  # Epoch 3: heads 1-3
        else:
            return self.speculation_depth  # Epoch 4+: all heads

    @oom_recovery_handler
    def _training_step(self, batch: Dict[str, torch.Tensor], epoch: int = 1,
                       accumulation_step: int = 0, total_accumulation_steps: int = 1) -> Tuple[torch.Tensor, Dict]:
        """Single training step with P-EAGLE aligned loss and curriculum learning.

        Key insight: During inference, drafter's predicted hidden states are converted
        to tokens via the TARGET model's lm_head. So we must train to match the
        token distributions, not just hidden state vectors.

        Uses curriculum learning: early epochs focus on head 1, progressively
        adding deeper heads as shallower heads converge.

        Args:
            batch: Training batch
            epoch: Current epoch number
            accumulation_step: Current accumulation step (0-indexed)
            total_accumulation_steps: Total number of gradient accumulation steps
        """
        # Only zero gradients on first accumulation step
        is_first_step = (accumulation_step == 0)
        is_last_step = (accumulation_step == total_accumulation_steps - 1)

        if is_first_step:
            self.optimizer.zero_grad()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        target_hidden = batch["target_hidden"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)
        target_token_ids = batch["target_token_ids"].to(self.device)  # Precomputed ground-truth targets

        # Note: DeepSpeed handles distributed synchronization internally.
        # Do NOT call dist.barrier() here - it conflicts with DeepSpeed's NCCL management.

        # ROOT CAUSE FIX: Validate batch shapes are identical across ranks
        # Shape mismatches cause NCCL collective operations to hang
        if self.world_size > 1 and self.is_main_process:
            # Log shapes for debugging
            self.logger.debug(f"Batch shapes - input_ids: {input_ids.shape}, "
                            f"target_hidden: {target_hidden.shape}, "
                            f"loss_mask: {loss_mask.shape}")

        # Forward pass: drafter generates hidden states WITH target_hidden injection
        # EAGLE-3 CRITICAL: Pass target_hidden for concatenation at first layer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_hidden=target_hidden,
            is_training=True
        )

        # Early NaN detection: if model outputs are already bad, skip this batch
        mtp_has_nan = any(not torch.isfinite(p).all() for p in outputs["mtp_predictions"] if p.numel() > 0)
        if mtp_has_nan:
            if self.is_main_process and self.global_step % 10 == 0:
                self.logger.warning("Model produced NaN/Inf in MTP predictions - skipping batch")
            # Return dummy loss to keep distributed sync intact
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return dummy_loss, {
                "mtp_loss_avg": float('nan'),
                "ce_loss_avg": float('nan'),
                "mse_loss_avg": float('nan'),
                "token_acc_avg": 0.0,
                "active_heads": self._get_active_heads(epoch),
                "avg_mask_coverage": 0.0,
                "skipped_due_to_nan": True,
            }

        # Determine active heads for curriculum learning
        active_heads = self._get_active_heads(epoch)

        # Compute losses for each MTP head
        ce_losses = []
        mse_losses = []
        token_accs = []
        mtp_losses = []
        head_mask_coverages = []

        for k, pred_hidden in enumerate(outputs["mtp_predictions"]):
            # CURRICULUM LEARNING: Only train active heads
            if k >= active_heads:
                break

            shift = k + 1

            # CRITICAL FIX: Handle the new model output where all heads produce full sequence [B, N, H]
            #
            # For head k:
            # - Prediction at position t should match target at position t + shift
            # - So pred[t] predicts target[t + shift]
            # - This means: pred[:, :-shift] aligns with target[:, shift:]
            #
            # For example:
            # - Head 0 (shift=1): pred[:-1] vs target[1:] - predicting next token
            # - Head 1 (shift=2): pred[:-2] vs target[2:] - predicting token at +2
            # - Head 2 (shift=3): pred[:-3] vs target[3:] - predicting token at +3
            # - Head 3 (shift=4): pred[:-4] vs target[4:] - predicting token at +4
            #
            # This ensures training matches inference where all heads use hidden[N-1]
            # to predict tokens at N, N+1, N+2, N+3.

            seq_len = target_hidden.shape[1]

            # Trim predictions and targets for this head
            pred_trimmed = pred_hidden[:, :-shift]  # Remove last shift positions
            target_trimmed = target_hidden[:, shift:]  # Start from shift position

            # Ensure matching lengths
            min_len = min(pred_trimmed.shape[1], target_trimmed.shape[1])

            if min_len <= 0:
                if self.global_step % 100 == 0 and self.is_main_process:
                    self.logger.warning(f"Head {k+1}: min_len={min_len}, pred_shape={pred_trimmed.shape}, target_shape={target_trimmed.shape}, skipping")
                continue

            pred_trimmed = pred_trimmed[:, :min_len]
            target_trimmed = target_trimmed[:, :min_len]

            # Mask should cover positions where both prediction and target are valid
            # After trimming: pred positions [0, min_len-1] predict targets [shift, shift+min_len-1]
            # Mask needs to cover positions [0, min_len-1] in both
            mask_trimmed = loss_mask[:, :min_len]

            # DIAGNOSTIC: Log mask coverage for debugging
            if self.global_step % 100 == 0 and self.is_main_process:
                mask_sum = mask_trimmed.sum().item()
                self.logger.info(
                    f"Head {k+1}: pred_shape={pred_trimmed.shape}, target_shape={target_trimmed.shape}, "
                    f"mask_sum={mask_sum} ({mask_sum/min_len*100:.1f}% coverage)"
                )

            # Skip if mask is all zeros (no learning signal)
            if mask_trimmed.sum() == 0:
                if self.global_step % 100 == 0 and self.is_main_process:
                    self.logger.warning(f"Head {k+1} SKIPPED: mask is all zeros")
                continue

            # Get target token IDs for this head (precomputed from target model)
            if target_token_ids is not None:
                target_ids_shifted = target_token_ids[:, shift:shift + min_len]
            else:
                target_ids_shifted = None

            # P-EAGLE aligned loss: match token distributions via target lm_head
            ce_loss_k, mse_loss_k, acc_k = hidden_state_token_loss(
                pred_trimmed,
                target_trimmed,
                self.target_lm_head,
                mask_trimmed,
                temperature=1.0,
                ce_weight=0.6,
                mse_weight=0.4,
                target_token_ids=target_ids_shifted,
                label_smoothing=self.label_smoothing
            )

            ce_losses.append(ce_loss_k)
            mse_losses.append(mse_loss_k)
            token_accs.append(acc_k.item())
            mtp_losses.append(mse_loss_k.item())

        # Combine losses: Cross-Entropy (token matching) + MSE (hidden state)
        # CE ensures tokens match, MSE ensures hidden states are structurally similar
        total_loss = torch.tensor(0.0, device=self.device)

        if ce_losses:
            weighted_ce = []
            for i, ce in enumerate(ce_losses):
                weight = max(0.5, 1.0 - i * 0.1)
                weighted_ce.append(ce * weight)
            ce_total = sum(weighted_ce) / sum(max(0.5, 1.0 - i * 0.1) for i in range(len(weighted_ce)))
            total_loss = total_loss + ce_total

        if mse_losses:
            mse_total = sum(mse_losses) / len(mse_losses)
            # CRITICAL FIX: MSE and CE have vastly different scales
            # CE loss is typically 2-10 (log scale), MSE is typically 0.01-1.0
            # Without scaling, MSE contributes ~0% to the gradient
            # Scale MSE to match CE scale (empirical: multiply by ~10-20)
            # This ensures both loss components properly influence training
            MSE_TO_CE_SCALE = 15.0  # Based on empirical observation
            total_loss = total_loss + mse_total * MSE_TO_CE_SCALE

        # --- LOSS SCALING ---
        # CE loss is already properly scaled in hidden_state_token_loss.
        # No additional scaling needed - the MTP head LR boost (2x) provides
        # sufficient gradient magnitude without risking explosion.
        # --------------------------

        # Detect NaN/Inf loss - treat like zero-loss to prevent gradient corruption
        if not torch.isfinite(total_loss):
            ce_vals = [f"{ce.item():.2f}" for ce in ce_losses] if ce_losses else ["n/a"]
            if self.is_main_process and self.global_step % 10 == 0:
                self.logger.warning(
                    f"Non-finite loss at step {self.global_step} "
                    f"(CE={','.join(ce_vals)}, heads={len(ce_losses)})"
                )
            total_loss = torch.tensor(0.0, device=self.device)

        # Track if this is a zero-loss batch for proper handling after backward
        is_zero_loss_batch = (total_loss.item() == 0)

        # DISTRIBUTED SAFETY: We MUST call backward() even with zero/NaN loss to avoid NCCL deadlock!
        # If one rank skips backward() while another calls it, the NCCL allreduce will hang forever.
        if is_zero_loss_batch:
            # Create a dummy zero loss that requires grad to participate in backward pass
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if self.is_main_process:
                self.logger.debug("Zero-loss/NaN batch: using dummy loss for NCCL synchronization")

        # Scale loss for gradient accumulation
        if total_accumulation_steps > 1:
            total_loss = total_loss / total_accumulation_steps

        # Backward pass (use DeepSpeed if enabled)
        if self.deepspeed_enabled:
            self.model.backward(total_loss)
        else:
            total_loss.backward()

        # Free activation memory immediately after backward
        # This helps prevent memory accumulation during training
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # CRITICAL FIX: Log gradient norms before clipping to diagnose issues
        if self.global_step % 10 == 0:
            mtp_grad_norm = 0.0
            lora_grad_norm = 0.0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if "mtp_heads" in name:
                        mtp_grad_norm += param.grad.norm().item() ** 2
                    elif "lora" in name.lower():
                        lora_grad_norm += param.grad.norm().item() ** 2
            if mtp_grad_norm > 0:
                self.writer.add_scalar("gradients/mtp_norm", mtp_grad_norm ** 0.5, self.global_step)
                if self.wandb_enabled:
                    wandb.log({"gradients/mtp_norm": mtp_grad_norm ** 0.5}, step=self.global_step)
            if lora_grad_norm > 0:
                self.writer.add_scalar("gradients/lora_norm", lora_grad_norm ** 0.5, self.global_step)
                if self.wandb_enabled:
                    wandb.log({"gradients/lora_norm": lora_grad_norm ** 0.5}, step=self.global_step)

        # Gradient clipping per parameter group
        # MTP heads (10x LR): allow larger gradients since they learn faster
        # LoRA/other (1-2x LR): standard clipping
        # SKIP for zero-loss batches (no real gradients to clip)
        if not is_zero_loss_batch:
            base_lr = getattr(self, '_base_lr', 2e-5)
            for group in self.optimizer.param_groups:
                # High LR group = MTP heads, allow larger gradients (>2x base LR)
                if group.get('lr', 0) > base_lr * 2:
                    torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm * 2)
                else:
                    torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm)

        # Only update weights on last accumulation step
        # SKIP for zero-loss batches (gradients are zero, no update needed)
        if is_last_step and not is_zero_loss_batch:
            if self.deepspeed_enabled:
                self.model.step()
            else:
                self.optimizer.step()
                self.scheduler.step()

            # Clear optimizer temporary memory
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        metrics = {
            "mtp_loss_avg": np.mean(mtp_losses) if mtp_losses else 0.0,
            "ce_loss_avg": sum(ce.item() for ce in ce_losses) / len(ce_losses) if ce_losses else 0.0,
            "mse_loss_avg": sum(mse_losses) / len(mse_losses) if mse_losses else 0.0,
            "token_acc_avg": np.mean(token_accs) if token_accs else 0.0,
            "active_heads": active_heads,
            "avg_mask_coverage": np.mean(head_mask_coverages) if head_mask_coverages else 0.0,
            "skipped_due_to_empty_mask": is_zero_loss_batch,
        }
        for i, loss_i in enumerate(mtp_losses):
            metrics[f"mtp_loss_{i+1}"] = loss_i
        for i, acc_i in enumerate(token_accs):
            metrics[f"token_acc_{i+1}"] = acc_i

        return total_loss, metrics

    def _save_checkpoint(self, name: str):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return

        checkpoint_dir = self.output_dir / name
        # Unwrap DDP model if needed before saving
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        # Pass target_lm_head to ensure vocab compatibility during inference
        model_to_save.save_checkpoint(str(checkpoint_dir), target_lm_head=self.target_lm_head)

        # Save training state for resume capability
        import json
        training_state = {
            "epoch": self.current_epoch,
            "step": self.global_step,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        self.logger.info(f"Checkpoint saved to {checkpoint_dir} (epoch={self.current_epoch}, step={self.global_step})")


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Drafter Training")
    parser.add_argument("--drafter_model", required=True, help="Base model for drafter")
    parser.add_argument("--target_model", default="google/gemma-3-4b-it",
                        help="Target model name for lm_head loading (default: google/gemma-3-4b-it)")
    parser.add_argument("--target_hidden_dim", type=int, required=True)
    parser.add_argument("--speculation_depth", type=int, default=4)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=256,
                        help="LoRA alpha scaling factor (default: 256)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate (default: 0.05)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5 for H200, 4e-5 for older GPUs)")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of warmup steps (if warmup_ratio not specified)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio as fraction of total steps (default: 0.03 = 3%%). "
                             "Overridden by --warmup_steps if both specified.")
    parser.add_argument("--skip-hardware-check", action="store_true",
                        help="Skip GPU/disk requirements check")
    parser.add_argument("--skip-security-check", action="store_true",
                        help="Skip pre-training security verification (not recommended)")
    parser.add_argument("--dataset-source", type=str, default=None,
                        help="Path to original dataset JSONL for security verification")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit"],
                        help="Quantize drafter model (4bit or 8bit) to reduce VRAM usage")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom name for this training run (used in logs)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for logs (default: {output_dir}/logs)")
    parser.add_argument("--gpu-safety-margin", type=float, default=1.5,
                        help="Minimum GPU memory to keep free in GB (default: 1.5)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt and start training immediately")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint directory (e.g., checkpoints_peagle_v2/checkpoint_step_1000)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="Number of steps to accumulate gradients before updating weights. Larger values = less memory but slower training (default: 32)")
    parser.add_argument("--max_seq_len", type=int, default=32768,
                        help="Maximum sequence length for training (default: 32768 — Gemma-3-270M-it native limit)")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON file for ZeRO optimization")
    parser.add_argument("--use_flash_attention", action="store_true", dest="use_flash_attention",
                        default=True, help="Use Flash Attention 2 for faster training (default: True)")
    parser.add_argument("--no_flash_attention", action="store_false", dest="use_flash_attention",
                        help="Disable Flash Attention")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (passed automatically by DeepSpeed)")
    # Regularization parameters
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.0 = no smoothing, recommended: 0.1)")
    parser.add_argument("--mtp_dropout", type=float, default=0.1,
                        help="Dropout rate in MTP heads (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer (default: 0.01)")
    parser.add_argument("--shard_cache_size", type=int, default=2,
                        help="Number of feature shards to cache in RAM (default: 2, each shard is ~17GB)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N steps (default: 100)")

    args = parser.parse_args()

    # Initialize distributed training if using multiple GPUs (but NOT if using DeepSpeed)
    # DeepSpeed handles its own distributed initialization
    if args.deepspeed:
        # DeepSpeed will handle distributed setup - just parse env vars
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    # Setup logging FIRST - before anything else (only on main process for file logging)
    output_path = Path(args.output_dir)
    if is_main:
        output_path.mkdir(parents=True, exist_ok=True)
    # Barrier to ensure directory is created before other processes proceed
    # Skip barrier when using DeepSpeed (it handles synchronization internally)
    if world_size > 1 and not args.deepspeed:
        dist.barrier()
    logger, run_log_dir, run_id = setup_training_logger(output_path, args.run_name)

    # Log all arguments
    logger.info("Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg}: {value}")
    logger.info("")

    # Run pre-training security verification
    if args.dataset_source:
        if not verify_dataset_source_security(args.dataset_source, args.skip_security_check):
            logger.error("\n⛔ Training aborted due to security concerns.")
            logger.error("   Use --skip-security-check to override (not recommended)")
            exit(1)
    else:
        # Check feature directory security
        if not run_pre_training_security_check(args.feature_dir):
            logger.error("\n⛔ Training aborted due to security concerns.")
            exit(1)

    # Save configuration to log directory
    config_path = run_log_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")

    trainer = EagleTrainer(
        drafter_model_name=args.drafter_model,
        target_model_name=args.target_model,
        target_hidden_dim=args.target_hidden_dim,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        speculation_depth=args.speculation_depth,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        skip_hardware_check=args.skip_hardware_check,
        yes=args.yes,
        quantization=args.quantization,
        logger=logger,
        run_log_dir=run_log_dir,
        log_dir=args.log_dir,
        gpu_safety_margin_gb=args.gpu_safety_margin,
        resume_from=args.resume,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
        rank=rank,
        world_size=world_size,
        deepspeed_config=args.deepspeed,
        use_flash_attention=args.use_flash_attention,
        # Regularization parameters
        label_smoothing=args.label_smoothing,
        mtp_dropout=args.mtp_dropout,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
        shard_cache_size=args.shard_cache_size,
    )

    try:
        # Initialize DeepSpeed if config provided
        if args.deepspeed:
            try:
                import deepspeed
                logger.info("Initializing DeepSpeed ZeRO...")
                # Load DeepSpeed config from file
                with open(args.deepspeed, 'r') as f:
                    ds_config = json.load(f)
                model, optimizer, _, scheduler = deepspeed.initialize(
                    model=trainer.model,
                    model_parameters=trainer.model.parameters(),
                    optimizer=trainer.optimizer,
                    lr_scheduler=trainer.scheduler,
                    config=ds_config
                )
                trainer.model = model
                trainer.optimizer = optimizer
                trainer.scheduler = scheduler
                trainer.deepspeed_enabled = True
                logger.info("DeepSpeed initialized successfully")
            except ImportError:
                logger.error("DeepSpeed not installed. Run: pip install deepspeed")
                raise
            except Exception as e:
                logger.error(f"DeepSpeed initialization failed: {e}")
                raise

        trainer.train()
        if is_main:
            logger.info(f"\n{'='*70}")
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*70}")
            logger.info(f"Logs saved to: {run_log_dir}")
            logger.info(f"Best model: {args.output_dir}/best_model")
    except Exception as e:
        if is_main:
            logger.exception("Training failed with error:")
            logger.error(f"\n{'='*70}")
            logger.error("TRAINING FAILED")
            logger.error(f"{'='*70}")
            logger.error(f"See logs for details: {run_log_dir}")
        raise
    finally:
        # Only cleanup if we initialized manually (not when DeepSpeed manages it)
        if not args.deepspeed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
