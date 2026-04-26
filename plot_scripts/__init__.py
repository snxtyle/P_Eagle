"""
P-EAGLE Plotting Scripts - Essential plots only

Models: Gemma 2B IT (drafter) -> Gemma 7B (target)
"""

from .plot_training import plot_training_loss
from .plot_evaluation import plot_acceptance_and_speedup
from .plot_comparison import plot_two_model_comparison

__all__ = [
    'plot_training_loss',
    'plot_acceptance_and_speedup',
    'plot_two_model_comparison',
]
