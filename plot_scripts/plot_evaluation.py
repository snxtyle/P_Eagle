"""
Plotting functions for evaluation metrics.
"""

from typing import Dict, Optional
from .utils import load_evaluation_results

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_acceptance_and_speedup(eval_file: str, output_path: str):
    """Single plot showing acceptance rates and speedup."""
    if not MATPLOTLIB_AVAILABLE:
        return

    results = load_evaluation_results(eval_file)
    if not results:
        return

    peagle = results.get('peagle', {})
    acceptance = peagle.get('acceptance_by_head', {})
    samples = peagle.get('samples', [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Acceptance rates
    ax1 = axes[0]
    positions = list(range(1, len(acceptance) + 1))
    rates = [acceptance.get(str(i), 0) * 100 for i in positions]

    colors = ['#27ae60', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax1.bar(positions, rates, color=colors[:len(rates)], edgecolor='black')
    ax1.set_xlabel('Head Position')
    ax1.set_ylabel('Acceptance Rate (%)')
    ax1.set_title('Token Acceptance (Gemma 2B IT Drafter)')
    ax1.set_ylim([0, 105])

    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', fontsize=10)

    # Right: Speedup per sample
    ax2 = axes[1]
    if samples:
        sample_nums = list(range(1, len(samples) + 1))
        speedups = [s.get('speedup_vs_naive', 0) for s in samples]

        ax2.bar(sample_nums, speedups, color='#3498db', edgecolor='black')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (AR)')
        ax2.set_xlabel('Test Sample')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Speedup vs Autoregressive')
        ax2.legend()

        avg_speedup = np.mean(speedups)
        ax2.text(0.98, 0.95, f'Mean: {avg_speedup:.2f}x',
                transform=ax2.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
