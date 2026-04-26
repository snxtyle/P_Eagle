"""
Utility functions for loading and processing metrics data.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def load_tensorboard_scalars(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """Load scalar metrics from TensorBoard logs."""
    if not TENSORBOARD_AVAILABLE:
        return {}

    scalars = {}
    log_path = Path(log_dir)

    if not log_path.exists():
        return {}

    event_files = list(log_path.glob("events.out.tfevents.*"))

    if not event_files:
        return {}

    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()

            for tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                scalars[tag] = [(e.step, e.value) for e in events]
        except Exception as e:
            print(f"Warning: Could not read {event_file}: {e}")

    return scalars


def load_evaluation_results(eval_file: str) -> Optional[Dict]:
    """Load evaluation results from JSON file."""
    if not os.path.exists(eval_file):
        return None

    with open(eval_file, 'r') as f:
        return json.load(f)


def load_checkpoint_config(checkpoint_dir: str) -> Optional[Dict]:
    """Load checkpoint configuration."""
    config_path = Path(checkpoint_dir) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def find_log_dirs(base_dirs: List[str]) -> Dict[str, str]:
    """Find TensorBoard log directories from checkpoint directories."""
    log_dirs = {}

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue

        # Check for logs subdirectory
        logs_subdir = base_path / 'logs'
        if logs_subdir.exists() and list(logs_subdir.glob("events.out.tfevents.*")):
            log_dirs[base_path.name] = str(logs_subdir)
            continue

        # Check parent directory for logs
        parent_logs = base_path.parent / 'logs'
        if parent_logs.exists() and list(parent_logs.glob("events.out.tfevents.*")):
            log_dirs[base_path.name] = str(parent_logs)

    return log_dirs


def smooth_curve(values: List[float], window: int = 10) -> List[float]:
    """Apply moving average smoothing to a curve."""
    if len(values) < window:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def format_model_name(name: str) -> str:
    """Format model name for display."""
    # Remove common prefixes/suffixes
    name = name.replace('checkpoints_', '')
    name = name.replace('checkpoint_', '')
    name = name.replace('_', ' ')
    return name.title()


def get_color_palette(n: int) -> List[str]:
    """Get a color palette for n items."""
    # Distinct color palette
    base_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]

    if n <= len(base_colors):
        return base_colors[:n]

    # Extend with matplotlib colormap if needed
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        return [cmap(i / n) for i in range(n)]
    except ImportError:
        return base_colors * ((n // len(base_colors)) + 1)[:n]
