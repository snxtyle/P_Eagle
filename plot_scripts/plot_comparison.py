"""
Plotting functions for comparing two models.
"""

from typing import Dict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_two_model_comparison(
    model1_name: str, model1: Dict,
    model2_name: str, model2: Dict,
    output_path: str
):
    """Simple side-by-side comparison of two models."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # MAL comparison
    ax1 = axes[0]
    models = [model1_name, model2_name]
    mals = [model1.get('mean_mal', 0), model2.get('mean_mal', 0)]

    bars = ax1.bar(models, mals, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax1.set_ylabel('Mean Acceptance Length')
    ax1.set_title('MAL Comparison')
    ax1.set_ylim([0, 4.5])
    ax1.axhline(y=2.5, color='green', linestyle='--', alpha=0.5)

    for bar, mal in zip(bars, mals):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{mal:.2f}', ha='center', fontsize=11, fontweight='bold')

    # Speedup comparison
    ax2 = axes[1]
    speedups = [model1.get('speedup_vs_baseline', 0), model2.get('speedup_vs_baseline', 0)]

    bars = ax2.bar(models, speedups, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup vs Autoregressive')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    for bar, sp in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{sp:.2f}x', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
