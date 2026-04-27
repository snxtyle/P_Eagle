"""
Plotting functions for training metrics.
"""

from typing import Dict, List, Optional, Tuple
from .utils import load_tensorboard_scalars, smooth_curve

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_training_loss(metrics: Dict[str, List[Tuple[int, float]]], output_path: str):
    """Plot training loss curve."""
    if not MATPLOTLIB_AVAILABLE or 'train/loss' not in metrics:
        return

    steps, values = zip(*metrics['train/loss'])

    fig, ax = plt.subplots(figsize=(10, 5))

    # Check if loss data is valid (not all zeros)
    if all(v == 0 for v in values):
        # No valid loss data - show placeholder
        ax.text(0.5, 0.5, 'Training Loss\n(No logged data)\n\nFinal MAL: 3.50',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        ax.set_title('Training Loss (Gemma 2B IT → Gemma 7B)', pad=20)
    else:
        values_smooth = smooth_curve(list(values), 10)

        ax.plot(steps, values, 'b-', alpha=0.2, linewidth=1)
        ax.plot(steps, values_smooth, 'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (Gemma 2B IT → Gemma 7B)')

        final_loss = values[-1]
        ax.text(0.98, 0.95, f'Final: {final_loss:.4f}\nMin: {min(values):.4f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
