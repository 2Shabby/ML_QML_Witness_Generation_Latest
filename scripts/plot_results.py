#!/usr/bin/env python3
"""
Plotting Script for ML-QML Witness Generation Experiments

Provides visualization for critical experimental flows:
1. Ablation study: 36D vs 63D feature comparison
2. Cross-validation: Multi-seed statistical validation
3. Per-family analysis: Accuracy by quantum state type
4. Noise robustness: Accuracy vs noise level curves
5. Witness coefficients: Pauli term importance analysis
6. Model comparison: SVM vs Transformer (if available)

Usage:
    python scripts/plot_results.py --plot ablation --results-file results/ablation_study_*.json
    python scripts/plot_results.py --plot all --results-dir results/
    python scripts/plot_results.py --plot noise --generate  # Generate and plot
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    DEFAULT_LOG_FORMAT,
    RESULTS_DIR,
    PROJECT_ROOT,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not installed. Install with: pip install matplotlib")

# Figure output directory
FIGURES_DIR = PROJECT_ROOT / 'figures'


def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_results(pattern: str, results_dir: Path = RESULTS_DIR, silent: bool = False) -> Optional[Dict]:
    """
    Load the most recent results file matching a pattern.

    Args:
        pattern: Glob pattern for results files (e.g., 'ablation_study_*.json')
        results_dir: Directory containing results
        silent: If True, don't log warnings for missing files

    Returns:
        Loaded results dictionary or None
    """
    files = sorted(glob(str(results_dir / pattern)))
    if not files:
        if not silent:
            logger.warning(f"No files matching '{pattern}' found in {results_dir}")
        return None

    latest_file = files[-1]
    logger.info(f"Loading: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)


def load_results_file(filepath: str) -> Dict:
    """Load results from a specific file."""
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# PLOT 1: Ablation Study - 36D vs 63D Features
# =============================================================================

def plot_ablation_study(results: Dict, save_path: Optional[Path] = None):
    """
    Plot ablation study comparing 36D restricted vs 63D full features.

    Creates a grouped bar chart showing accuracy, precision, recall, F1
    for both feature sets. Handles both SVM-only and multi-model results.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    # Check if this is multi-model comparison (transformer) or single-model (SVM)
    is_multi_model = 'models' in results

    if is_multi_model:
        # Multi-model ablation comparison
        models = list(results['models'].keys())
        n_models = len(models)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: 36D vs 63D accuracy by model
        ax1 = axes[0]

        x = np.arange(n_models)
        width = 0.35

        acc_36d = [results['models'][m]['restricted_36d']['accuracy_mean'] for m in models]
        acc_63d = [results['models'][m]['full_63d']['accuracy_mean'] for m in models]
        std_36d = [results['models'][m]['restricted_36d']['accuracy_std'] for m in models]
        std_63d = [results['models'][m]['full_63d']['accuracy_std'] for m in models]

        bars1 = ax1.bar(x - width/2, acc_36d, width, yerr=std_36d, label='36D Restricted',
                        color='#2ecc71', alpha=0.8, capsize=5)
        bars2 = ax1.bar(x + width/2, acc_63d, width, yerr=std_63d, label='63D Full',
                        color='#3498db', alpha=0.8, capsize=5)

        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('36D vs 63D Feature Comparison (All Models)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in models], rotation=15, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)

        # Right plot: Accuracy gap by model
        ax2 = axes[1]

        gaps = [results['models'][m]['statistical_comparison']['accuracy_gap'] for m in models]
        p_vals = [results['models'][m]['statistical_comparison']['paired_ttest_p'] for m in models]
        sig = [results['models'][m]['statistical_comparison']['significant_at_0.05'] for m in models]

        colors = ['#e74c3c' if s else '#95a5a6' for s in sig]
        bars = ax2.bar(models, gaps, color=colors, alpha=0.8, edgecolor='black')

        # Add p-value annotations
        for bar, p, s in zip(bars, p_vals, sig):
            marker = '*' if s else ''
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'p={p:.3f}{marker}', ha='center', va='bottom', fontsize=9)

        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Accuracy Gap (63D - 36D)')
        ax2.set_title('Feature Dimension Effect by Model')
        ax2.set_xticklabels([m.upper() for m in models], rotation=15, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        # Add legend for significance
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e74c3c', label='Significant (p<0.05)'),
                          Patch(facecolor='#95a5a6', label='Not significant')]
        ax2.legend(handles=legend_elements, loc='upper right')

    else:
        # Single-model (SVM) ablation plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left plot: Accuracy comparison with error bars
        ax1 = axes[0]

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metrics))
        width = 0.35

        restricted = results['restricted_36d']
        full = results['full_63d']

        values_36d = [
            restricted['accuracy_mean'],
            restricted['precision_mean'],
            restricted['recall_mean'],
            restricted['f1_mean']
        ]
        values_63d = [
            full['accuracy_mean'],
            full['precision_mean'],
            full['recall_mean'],
            full['f1_mean']
        ]

        bars1 = ax1.bar(x - width/2, values_36d, width, label='36D Restricted', color='#2ecc71', alpha=0.8)
        bars2 = ax1.bar(x + width/2, values_63d, width, label='63D Full', color='#3498db', alpha=0.8)

        # Add error bar for accuracy
        ax1.errorbar(x[0] - width/2, values_36d[0], yerr=restricted['accuracy_std'],
                     fmt='none', color='black', capsize=5)
        ax1.errorbar(x[0] + width/2, values_63d[0], yerr=full['accuracy_std'],
                     fmt='none', color='black', capsize=5)

        ax1.set_ylabel('Score')
        ax1.set_title('36D Restricted vs 63D Full Features')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)

        # Right plot: Fold-by-fold accuracy comparison
        ax2 = axes[1]

        folds = range(1, len(restricted['fold_accuracies']) + 1)
        ax2.plot(folds, restricted['fold_accuracies'], 'o-', label='36D Restricted',
                 color='#2ecc71', linewidth=2, markersize=8)
        ax2.plot(folds, full['fold_accuracies'], 's-', label='63D Full',
                 color='#3498db', linewidth=2, markersize=8)

        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Cross-Validation Fold Accuracies')
        ax2.legend()
        ax2.set_xticks(folds)
        ax2.grid(alpha=0.3)

        # Add statistical info
        stat = results['statistical_comparison']
        significance = "Significant" if stat['significant_at_0.05'] else "Not significant"
        fig.text(0.5, 0.02,
                 f"Gap: {stat['accuracy_gap']:.4f} | t-stat: {stat['paired_ttest_t']:.3f} | "
                 f"p-value: {stat['paired_ttest_p']:.4f} ({significance} at p<0.05)",
                 ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# PLOT 2: Cross-Validation Multi-Seed Results
# =============================================================================

def plot_cross_validation(results: Dict, save_path: Optional[Path] = None):
    """
    Plot cross-validation results across multiple random seeds.

    Shows distribution of accuracies and per-seed breakdown.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Box plot of accuracies
    ax1 = axes[0]

    all_accuracies = []
    for seed_result in results['per_seed_results']:
        all_accuracies.extend(seed_result['fold_accuracies'])

    bp = ax1.boxplot([all_accuracies], labels=['All Folds'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)

    # Overlay individual points
    x_jitter = np.random.normal(1, 0.04, len(all_accuracies))
    ax1.scatter(x_jitter, all_accuracies, alpha=0.5, color='#2c3e50', s=30)

    stats = results['aggregate_statistics']
    ax1.axhline(stats['accuracy_mean'], color='red', linestyle='--',
                label=f"Mean: {stats['accuracy_mean']:.4f}")

    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Distribution Across All Seeds/Folds')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right plot: Per-seed mean accuracy
    ax2 = axes[1]

    seeds = [r['seed'] for r in results['per_seed_results']]
    mean_accs = [r['accuracy'] for r in results['per_seed_results']]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(seeds)))
    bars = ax2.bar(range(len(seeds)), mean_accs, color=colors, alpha=0.8)

    ax2.axhline(stats['accuracy_mean'], color='red', linestyle='--', linewidth=2,
                label=f"Overall Mean: {stats['accuracy_mean']:.4f}")
    ax2.fill_between([-0.5, len(seeds)-0.5],
                     stats['accuracy_mean'] - stats['accuracy_std'],
                     stats['accuracy_mean'] + stats['accuracy_std'],
                     alpha=0.2, color='red', label=f"±1 Std: {stats['accuracy_std']:.4f}")

    ax2.set_xlabel('Seed')
    ax2.set_ylabel('Mean Accuracy')
    ax2.set_title('Mean Accuracy by Random Seed')
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels(seeds, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# PLOT 3: Per-Family Analysis
# =============================================================================

def plot_per_family(results: Dict, save_path: Optional[Path] = None):
    """
    Plot accuracy breakdown by quantum state family.

    Shows how well the classifier performs on each type of state.
    Handles both SVM-only results and multi-model comparison results.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    families = list(results['per_family'].keys())
    family_data = results['per_family']

    # Check if this is multi-model comparison (transformer) or single-model (SVM)
    first_family = family_data[families[0]]
    is_multi_model = 'models' in first_family

    if is_multi_model:
        # Multi-model comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: Grouped bar chart by family and model
        ax1 = axes[0]

        models = list(first_family['models'].keys())
        x = np.arange(len(families))
        width = 0.8 / len(models)
        model_colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']

        for i, model in enumerate(models):
            accuracies = [family_data[f]['models'][model]['accuracy'] for f in families]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax1.bar(x + offset, accuracies, width, label=model.upper(),
                          color=model_colors[i % len(model_colors)], alpha=0.8)

        ax1.set_xlabel('State Family')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy by State Family (All Models)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f.upper() for f in families])
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 1.15)
        ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax1.grid(axis='y', alpha=0.3)

        # Right plot: Heatmap of accuracies (models x families)
        ax2 = axes[1]

        heatmap_data = np.array([
            [family_data[f]['models'][m]['accuracy'] for f in families]
            for m in models
        ])

        im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1)

        ax2.set_xticks(range(len(families)))
        ax2.set_xticklabels([f.upper() for f in families])
        ax2.set_yticks(range(len(models)))
        ax2.set_yticklabels([m.upper() for m in models])
        ax2.set_title('Accuracy Heatmap: Models vs Families')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(families)):
                text = ax2.text(j, i, f'{heatmap_data[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=10)

        plt.colorbar(im, ax=ax2, label='Accuracy')

    else:
        # Single-model (SVM) plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Accuracy by family
        ax1 = axes[0]

        accuracies = [family_data[f]['accuracy'] for f in families]
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']

        bars = ax1.bar(families, accuracies, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy by State Family')
        ax1.set_ylim(0, 1.15)
        ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Capitalize family names
        ax1.set_xticklabels([f.upper() for f in families])

        # Right plot: Distillable fraction and metrics heatmap
        ax2 = axes[1]

        metrics_data = np.array([
            [family_data[f]['accuracy'] for f in families],
            [family_data[f]['precision'] for f in families],
            [family_data[f]['recall'] for f in families],
            [family_data[f]['distillable_fraction'] for f in families]
        ])

        im = ax2.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax2.set_xticks(range(len(families)))
        ax2.set_xticklabels([f.upper() for f in families])
        ax2.set_yticks(range(4))
        ax2.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'Distill. Frac.'])
        ax2.set_title('Performance Metrics Heatmap')

        # Add text annotations
        for i in range(4):
            for j in range(len(families)):
                text = ax2.text(j, i, f'{metrics_data[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=9)

        plt.colorbar(im, ax=ax2, label='Score')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# PLOT 4: Noise Robustness
# =============================================================================

def plot_noise_robustness(results: Dict, save_path: Optional[Path] = None):
    """
    Plot accuracy vs noise level curves.

    Shows how classifier performance degrades with increasing noise.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    noise_levels = sorted([float(k) for k in results['per_noise_level'].keys()])
    noise_data = results['per_noise_level']

    accuracies = [noise_data[str(n)]['accuracy'] for n in noise_levels]
    dist_fracs = [noise_data[str(n)]['distillable_fraction'] for n in noise_levels]

    # Left plot: Accuracy vs noise
    ax1 = axes[0]

    ax1.plot(noise_levels, accuracies, 'o-', color='#3498db', linewidth=2,
             markersize=10, label='Test Accuracy')
    ax1.fill_between(noise_levels, accuracies, alpha=0.3, color='#3498db')

    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Random baseline')

    # Find noise threshold where accuracy drops below 0.7
    threshold_idx = next((i for i, a in enumerate(accuracies) if a < 0.7), None)
    if threshold_idx:
        ax1.axvline(noise_levels[threshold_idx], color='red', linestyle='--',
                   alpha=0.7, label=f'Threshold (~0.7 acc)')

    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Classification Accuracy vs Noise Level')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-0.02, max(noise_levels) + 0.02)
    ax1.set_ylim(0.4, 1.05)

    # Right plot: Accuracy and distillable fraction together
    ax2 = axes[1]

    ax2.plot(noise_levels, accuracies, 'o-', color='#3498db', linewidth=2,
             markersize=8, label='Accuracy')
    ax2.plot(noise_levels, dist_fracs, 's--', color='#e74c3c', linewidth=2,
             markersize=8, label='Distillable Fraction')

    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Value')
    ax2.set_title('Accuracy vs Class Balance')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.02, max(noise_levels) + 0.02)
    ax2.set_ylim(0, 1.05)

    # Add annotation about noise effect
    ax2.annotate('Higher noise → fewer distillable states',
                xy=(0.5, 0.3), fontsize=9, style='italic', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# PLOT 5: Witness Coefficient Analysis
# =============================================================================

def plot_witness_coefficients(results: Dict, save_path: Optional[Path] = None):
    """
    Plot witness coefficient analysis showing Pauli term importance.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left plot: Top 15 Pauli coefficients
    ax1 = axes[0]

    top_coeffs = results['ranked_coefficients'][:15]
    paulis = [c['pauli'] for c in top_coeffs]
    values = [c['coefficient'] for c in top_coeffs]
    signs = [c['sign'] for c in top_coeffs]

    colors = ['#2ecc71' if s == '+' else '#e74c3c' for s in signs]

    y_pos = range(len(paulis))
    bars = ax1.barh(y_pos, values, color=colors, alpha=0.8)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(paulis, fontsize=9, fontfamily='monospace')
    ax1.set_xlabel('|Coefficient|')
    ax1.set_title('Top 15 Pauli Terms by Importance')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Add legend for sign
    green_patch = mpatches.Patch(color='#2ecc71', label='Positive')
    red_patch = mpatches.Patch(color='#e74c3c', label='Negative')
    ax1.legend(handles=[green_patch, red_patch], loc='lower right')

    # Middle plot: 1-body vs 2-body importance pie chart
    ax2 = axes[1]

    importance = results.get('importance_by_type', {})
    if importance:
        sizes = [importance.get('one_body_fraction', 0),
                 importance.get('two_body_fraction', 0)]
        labels = ['1-Body\n(Local)', '2-Body\n(Correlations)']
        colors = ['#3498db', '#e74c3c']
        explode = (0, 0.05)

        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels,
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90)
        ax2.set_title('Importance by Pauli Weight')
    else:
        ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center')
        ax2.set_title('Importance by Pauli Weight')

    # Right plot: Coefficient magnitude distribution
    ax3 = axes[2]

    all_coeffs = [c['coefficient'] for c in results['all_coefficients']]

    ax3.hist(all_coeffs, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(all_coeffs), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_coeffs):.4f}')

    ax3.set_xlabel('|Coefficient|')
    ax3.set_ylabel('Count')
    ax3.set_title('Coefficient Magnitude Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Add summary statistics
    stats = results['witness_statistics']
    fig.text(0.5, 0.02,
             f"Total terms: {stats['n_terms']} | "
             f"1-body: {stats['n_one_body']} | 2-body: {stats['n_two_body']} | "
             f"Max coeff: {stats['max_abs_coefficient']:.4f}",
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# PLOT 6: Model Comparison (SVM vs Transformer)
# =============================================================================

def plot_model_comparison(results: Dict, save_path: Optional[Path] = None):
    """
    Plot comparison between SVM and Transformer models.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    models = results.get('models', {})
    if not models:
        logger.warning("No model comparison data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Metrics comparison
    ax1 = axes[0]

    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

    model_names = list(models.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']

    for i, (model_name, model_data) in enumerate(models.items()):
        if 'error' in model_data:
            continue
        values = [model_data.get(m, 0) for m in metrics]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax1.bar(x + offset, values, width, label=model_name.upper(),
                      color=colors[i % len(colors)], alpha=0.8)

    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    # Right plot: Parameters vs Accuracy trade-off
    ax2 = axes[1]

    valid_models = [(name, data) for name, data in models.items() if 'error' not in data]

    if valid_models:
        names = [name for name, _ in valid_models]
        accs = [data['test_accuracy'] for _, data in valid_models]
        params = [data.get('n_support_vectors', data.get('n_parameters', 100))
                  for _, data in valid_models]

        scatter = ax2.scatter(params, accs, s=200, c=range(len(names)),
                             cmap='viridis', alpha=0.8, edgecolors='black')

        for i, (name, acc, param) in enumerate(zip(names, accs, params)):
            ax2.annotate(name.upper(), (param, acc),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')

        ax2.set_xlabel('Model Complexity (Support Vectors / Parameters)')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Accuracy vs Model Complexity')
        ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    plt.show()


# =============================================================================
# COMPREHENSIVE DASHBOARD
# =============================================================================

def plot_all_from_directory(results_dir: Path, save_figures: bool = True):
    """
    Generate all available plots from results in a directory.

    Uses a category-based approach: for each experiment type, tries multiple
    patterns in order (transformer comparison first, then SVM-only fallback).
    Only warns once per category if no results found.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    ensure_figures_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define experiment categories with alternative patterns (try in order)
    # Each category has: (category_name, [(pattern, plot_name), ...], plot_function)
    experiment_categories = [
        ('ablation', [
            ('ablation_comparison_*.json', 'ablation_all_models'),  # Transformer first
            ('ablation_study_*.json', 'ablation_svm'),
        ], plot_ablation_study),
        ('cross_validation', [
            ('cv_comparison_*.json', 'cv_comparison_all_models'),
            ('cross_validation_*.json', 'cross_validation_svm'),
        ], plot_cross_validation),
        ('per_family', [
            ('per_family_comparison_*.json', 'per_family_all_models'),
            ('per_family_analysis_*.json', 'per_family_svm'),
        ], plot_per_family),
        ('noise_robustness', [
            ('noise_robustness_*.json', 'noise_robustness'),
        ], plot_noise_robustness),
        ('witness', [
            ('witness_analysis_*.json', 'witness_analysis_transformer'),
            ('witness_coefficients_*.json', 'witness_coefficients_svm'),
        ], plot_witness_coefficients),
        ('model_comparison', [
            ('model_comparison_*.json', 'model_comparison'),
        ], plot_model_comparison),
    ]

    plots_generated = 0
    categories_missing = []

    for category_name, patterns, plot_func in experiment_categories:
        # Try each pattern in order until we find results (silently)
        found = False
        for pattern, plot_name in patterns:
            results = load_latest_results(pattern, results_dir, silent=True)
            if results:
                save_path = FIGURES_DIR / f"{plot_name}_{timestamp}.png" if save_figures else None
                try:
                    plot_func(results, save_path)
                    plots_generated += 1
                    found = True
                    break  # Stop after first successful pattern in category
                except Exception as e:
                    logger.error(f"Error plotting {plot_name}: {e}")

        if not found:
            categories_missing.append(category_name)

    # Summary
    logger.info(f"Generated {plots_generated} plots")

    if categories_missing:
        logger.info(f"No results found for: {', '.join(categories_missing)}")

    if save_figures and plots_generated > 0:
        logger.info(f"Figures saved to: {FIGURES_DIR}")


# =============================================================================
# SUMMARY DASHBOARD (Single Figure)
# =============================================================================

def load_first_available(patterns: List[str], results_dir: Path):
    """Try multiple patterns in order and return the first results found (silently)."""
    for pattern in patterns:
        results = load_latest_results(pattern, results_dir, silent=True)
        if results:
            return results
    return None


def plot_summary_dashboard(results_dir: Path, save_path: Optional[Path] = None):
    """
    Create a single comprehensive dashboard with all key results.
    Tries transformer comparison results first, then falls back to SVM-only.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    fig = plt.figure(figsize=(16, 12))

    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Ablation study (top-left) - try transformer first, then SVM
    ax1 = fig.add_subplot(gs[0, 0])
    ablation = load_first_available(['ablation_comparison_*.json', 'ablation_study_*.json'], results_dir)
    if ablation:
        # Check if multi-model (transformer) or single-model (SVM)
        if 'models' in ablation:
            # Multi-model: show first model's 36D vs 63D
            first_model = list(ablation['models'].keys())[0]
            r36 = ablation['models'][first_model]['restricted_36d']
            r63 = ablation['models'][first_model]['full_63d']
            title = f'Ablation: 36D vs 63D ({first_model.upper()})'
        else:
            r36 = ablation['restricted_36d']
            r63 = ablation['full_63d']
            title = 'Ablation: 36D vs 63D'

        metrics = ['Acc', 'Prec', 'Rec', 'F1']
        x = np.arange(len(metrics))
        width = 0.35
        ax1.bar(x - width/2, [r36['accuracy_mean'], r36.get('precision_mean', 0),
                              r36.get('recall_mean', 0), r36.get('f1_mean', 0)],
               width, label='36D', color='#2ecc71', alpha=0.8)
        ax1.bar(x + width/2, [r63['accuracy_mean'], r63.get('precision_mean', 0),
                              r63.get('recall_mean', 0), r63.get('f1_mean', 0)],
               width, label='63D', color='#3498db', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.set_ylabel('Score')
        ax1.set_title(title)
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 1.1)

    # 2. Cross-validation (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    cv = load_first_available(['cv_comparison_*.json', 'cross_validation_*.json'], results_dir)
    if cv:
        if 'per_seed_results' in cv:
            # SVM format
            seeds = [r['seed'] for r in cv['per_seed_results']]
            accs = [r['accuracy'] for r in cv['per_seed_results']]
            mean_acc = cv['aggregate_statistics']['accuracy_mean']
        elif 'models' in cv:
            # Transformer CV format - show first model's fold accuracies
            first_model = list(cv['models'].keys())[0]
            accs = cv['models'][first_model]['fold_accuracies']
            seeds = list(range(1, len(accs) + 1))
            mean_acc = cv['models'][first_model]['accuracy_mean']
        else:
            seeds, accs, mean_acc = [], [], 0

        if accs:
            ax2.bar(range(len(seeds)), accs, color='#9b59b6', alpha=0.8)
            ax2.axhline(mean_acc, color='red', linestyle='--', label=f"Mean: {mean_acc:.3f}")
            ax2.set_xlabel('Fold/Seed Index')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Cross-Validation Stability')
            ax2.legend(fontsize=8)

    # 3. Per-family (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    family = load_first_available(['per_family_comparison_*.json', 'per_family_analysis_*.json'], results_dir)
    if family:
        families = list(family['per_family'].keys())
        first_family = family['per_family'][families[0]]

        # Check if multi-model
        if 'models' in first_family:
            # Multi-model: use SVM accuracy for dashboard simplicity
            accs = [family['per_family'][f]['models']['svm']['accuracy'] for f in families]
            title = 'Accuracy by State Family (SVM)'
        else:
            accs = [family['per_family'][f]['accuracy'] for f in families]
            title = 'Accuracy by State Family'

        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
        ax3.bar(families, accs, color=colors[:len(families)], alpha=0.8)
        ax3.set_ylabel('Accuracy')
        ax3.set_title(title)
        ax3.set_xticklabels([f.upper() for f in families], rotation=45, ha='right')

    # 4. Noise robustness (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    noise = load_latest_results('noise_robustness_*.json', results_dir, silent=True)
    if noise:
        levels = sorted([float(k) for k in noise['per_noise_level'].keys()])
        accs = [noise['per_noise_level'][str(n)]['accuracy'] for n in levels]
        ax4.plot(levels, accs, 'o-', color='#3498db', linewidth=2, markersize=6)
        ax4.fill_between(levels, accs, alpha=0.3, color='#3498db')
        ax4.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Noise Robustness')

    # 5. Witness coefficients - top terms (middle-center)
    ax5 = fig.add_subplot(gs[1, 1])
    witness = load_first_available(['witness_analysis_*.json', 'witness_coefficients_*.json'], results_dir)
    if witness:
        # Handle transformer witness format (has 'witnesses' key) or SVM format
        if 'witnesses' in witness:
            # Transformer format - use SVM witness for dashboard
            if 'svm' in witness['witnesses']:
                top = witness['witnesses']['svm'].get('top_coefficients', [])[:10]
                paulis = [c[0] for c in top]  # Format is [pauli, coeff]
                values = [abs(c[1]) for c in top]
                signs = ['+' if c[1] >= 0 else '-' for c in top]
            else:
                top, paulis, values, signs = [], [], [], []
        else:
            # SVM format
            top = witness.get('ranked_coefficients', [])[:10]
            paulis = [c['pauli'] for c in top]
            values = [c['coefficient'] for c in top]
            signs = [c['sign'] for c in top]

        if paulis:
            colors = ['#2ecc71' if s == '+' else '#e74c3c' for s in signs]
            ax5.barh(range(len(paulis)), values, color=colors, alpha=0.8)
            ax5.set_yticks(range(len(paulis)))
            ax5.set_yticklabels(paulis, fontsize=7, fontfamily='monospace')
            ax5.set_xlabel('|Coefficient|')
            ax5.set_title('Top 10 Pauli Terms')
            ax5.invert_yaxis()

    # 6. 1-body vs 2-body importance (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    if witness:
        imp = witness.get('importance_by_type', {})
        if imp:
            sizes = [imp.get('one_body_fraction', 0), imp.get('two_body_fraction', 0)]
            ax6.pie(sizes, labels=['1-Body', '2-Body'], colors=['#3498db', '#e74c3c'],
                   autopct='%1.1f%%', startangle=90)
            ax6.set_title('Importance by Weight')

    # 7. Model comparison (bottom, spanning full width)
    ax7 = fig.add_subplot(gs[2, :])
    comparison = load_latest_results('model_comparison_*.json', results_dir, silent=True)
    if comparison and 'models' in comparison:
        models = comparison['models']
        model_names = [k.upper() for k in models.keys() if 'error' not in models[k]]
        metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

        x = np.arange(len(metric_labels))
        width = 0.2
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        for i, model_name in enumerate(model_names[:3]):  # Max 3 models
            values = [models[model_name.lower()].get(m, 0) for m in metrics_to_plot]
            ax7.bar(x + i * width, values, width, label=model_name,
                   color=colors[i], alpha=0.8)

        ax7.set_xticks(x + width)
        ax7.set_xticklabels(metric_labels)
        ax7.set_ylabel('Score')
        ax7.set_title('Model Comparison: SVM vs Transformer')
        ax7.legend(fontsize=9)
        ax7.set_ylim(0, 1.1)
        ax7.grid(axis='y', alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Model comparison data not available',
                ha='center', va='center', fontsize=12)
        ax7.set_title('Model Comparison')

    plt.suptitle('ML-QML Witness Generation: Experimental Results Summary',
                fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved dashboard: {save_path}")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot experimental results for ML-QML Witness Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_results.py --plot ablation
  python scripts/plot_results.py --plot all --results-dir results/
  python scripts/plot_results.py --plot dashboard --save
  python scripts/plot_results.py --plot noise --results-file results/noise_robustness_20241217.json
        """
    )

    parser.add_argument(
        '--plot',
        type=str,
        required=True,
        choices=['ablation', 'cv', 'family', 'noise', 'witness', 'comparison',
                 'all', 'dashboard'],
        help='Which plot(s) to generate'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default=str(RESULTS_DIR),
        help=f'Directory containing results (default: {RESULTS_DIR})'
    )

    parser.add_argument(
        '--results-file',
        type=str,
        default=None,
        help='Specific results file to load'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save figures to figures/ directory'
    )

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    results_dir = Path(args.results_dir)
    ensure_figures_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Map plot types to patterns (try transformer comparison first, then SVM-only)
    plot_map = {
        'ablation': (['ablation_comparison_*.json', 'ablation_study_*.json'], plot_ablation_study),
        'cv': (['cv_comparison_*.json', 'cross_validation_*.json'], plot_cross_validation),
        'family': (['per_family_comparison_*.json', 'per_family_analysis_*.json'], plot_per_family),
        'noise': (['noise_robustness_*.json'], plot_noise_robustness),
        'witness': (['witness_analysis_*.json', 'witness_coefficients_*.json'], plot_witness_coefficients),
        'comparison': (['model_comparison_*.json'], plot_model_comparison),
    }

    if args.plot == 'all':
        plot_all_from_directory(results_dir, save_figures=args.save)

    elif args.plot == 'dashboard':
        save_path = FIGURES_DIR / f"dashboard_{timestamp}.png" if args.save else None
        plot_summary_dashboard(results_dir, save_path)

    else:
        patterns, plot_func = plot_map[args.plot]

        if args.results_file:
            results = load_results_file(args.results_file)
        else:
            # Try each pattern until we find results (silently)
            results = None
            for pattern in patterns:
                results = load_latest_results(pattern, results_dir, silent=True)
                if results:
                    break

        if results:
            save_path = FIGURES_DIR / f"{args.plot}_{timestamp}.png" if args.save else None
            plot_func(results, save_path)
        else:
            # Only show error with pattern list when nothing found
            logger.error(f"No results found for '{args.plot}'. Tried patterns: {patterns}")
            sys.exit(1)


if __name__ == '__main__':
    main()
