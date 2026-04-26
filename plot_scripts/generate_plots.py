#!/usr/bin/env python3
"""
P-EAGLE Plot Generator - Simplified

Generates essential plots for training and evaluation.
Models: Gemma 2B IT (drafter) -> Gemma 7B (target)

Usage:
    python -m plot_scripts.generate_plots

    # Compare two models
    python -m plot_scripts.generate_plots --mode compare \
        --model1 results/a.json --model2 results/b.json \
        --model1_name "Config A" --model2_name "Config B"
"""

import argparse
import os
from pathlib import Path

from .utils import load_tensorboard_scalars, load_evaluation_results, find_log_dirs
from .plot_training import plot_training_loss
from .plot_evaluation import plot_acceptance_and_speedup
from .plot_comparison import plot_two_model_comparison


def main():
    parser = argparse.ArgumentParser(description="Generate P-EAGLE plots")
    parser.add_argument('--mode', choices=['single', 'compare', 'all'], default='all')
    parser.add_argument('--checkpoint_dirs', nargs='+', default=['checkpoints_gemma'])
    parser.add_argument('--eval_file', default='evaluation_results.json')
    parser.add_argument('--model1', help='First model results')
    parser.add_argument('--model2', help='Second model results')
    parser.add_argument('--model1_name', default='Model 1')
    parser.add_argument('--model2_name', default='Model 2')
    parser.add_argument('--output_dir', default='plot_scripts/plots')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*50)
    print("P-EAGLE Plot Generator")
    print("Models: Gemma 2B IT → Gemma 7B")
    print("="*50)

    if args.mode in ['single', 'all']:
        # Training plot
        log_dirs = find_log_dirs(args.checkpoint_dirs)
        for name, log_dir in log_dirs.items():
            metrics = load_tensorboard_scalars(log_dir)
            if metrics:
                plot_training_loss(
                    metrics,
                    os.path.join(args.output_dir, 'training_loss.png')
                )

        # Evaluation plot
        if os.path.exists(args.eval_file):
            plot_acceptance_and_speedup(
                args.eval_file,
                os.path.join(args.output_dir, 'evaluation_metrics.png')
            )

    if args.mode == 'compare' and args.model1 and args.model2:
        m1 = load_evaluation_results(args.model1)
        m2 = load_evaluation_results(args.model2)
        if m1 and m2:
            plot_two_model_comparison(
                args.model1_name, m1.get('peagle', m1),
                args.model2_name, m2.get('peagle', m2),
                os.path.join(args.output_dir, 'model_comparison.png')
            )

    print(f"\nPlots saved to {args.output_dir}/")
    return 0


if __name__ == '__main__':
    exit(main())
