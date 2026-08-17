#!/usr/bin/env python3
"""
Comparative Analysis: SVM vs Transformer Models

Runs a comprehensive comparison of all 3 model architectures:
1. Linear SVM (baseline)
2. Transformer Classifier (pure classification)
3. Hybrid Transformer (with witness extraction)

Generates ML-appropriate metrics and visualizations:
- Confusion matrices
- ROC curves and AUC scores
- Precision-Recall curves
- Learning curves
- Coefficient/attention analysis

Usage:
    python scripts/run_comparative_analysis.py --n-samples 2000
    python scripts/run_comparative_analysis.py --n-samples 5000 --save-plots
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.svm import SVC

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quantum_states.state_generation import generate_distillability_dataset
from src.feature_extraction.pauli_features import (
    create_sparse_measurement_set,
    extract_features_batch
)
from src.config import (
    DEFAULT_SEED,
    DEFAULT_TRANSFORMER_CONFIG,
    DEFAULT_MLP_CONFIG,
    DEFAULT_LOG_FORMAT,
    RESULTS_DIR,
    PROJECT_ROOT,
)
from src.utils import convert_to_json_serializable, set_seed, TORCH_AVAILABLE

# Try to import transformer and MLP
if TORCH_AVAILABLE:
    from src.ml_models.transformer_witness import TransformerWitnessLearner
    from src.ml_models.mlp_classifier import MLPClassifierLearner

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
logger = logging.getLogger(__name__)

# Output directories
FIGURES_DIR = PROJECT_ROOT / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class ModelEvaluator:
    """Evaluate and compare multiple models with ML-standard metrics."""

    def __init__(self, X_train, y_train, X_test, y_test, basis, seed=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.basis = basis
        self.seed = seed
        self.results = {}

    def evaluate_svm(self) -> Dict:
        """Train and evaluate SVM model."""
        logger.info("Training SVM...")

        svm = SVC(kernel='linear', C=1.0, random_state=self.seed, probability=True)
        svm.fit(self.X_train, self.y_train)

        y_pred = svm.predict(self.X_test)
        y_proba = svm.predict_proba(self.X_test)[:, 1]

        self.results['svm'] = {
            'name': 'Linear SVM',
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metrics': self._compute_metrics(y_pred, y_proba),
            'n_support_vectors': len(svm.support_),
            'model': svm
        }

        logger.info(f"SVM Test Accuracy: {self.results['svm']['metrics']['accuracy']:.4f}")
        return self.results['svm']

    def evaluate_transformer_classifier(self) -> Optional[Dict]:
        """Train and evaluate Transformer Classifier."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping Transformer Classifier")
            return None

        logger.info("Training Transformer Classifier...")

        config = DEFAULT_TRANSFORMER_CONFIG.to_dict()
        learner = TransformerWitnessLearner(
            pauli_basis=self.basis,
            mode='classifier',
            **config,
            random_state=self.seed
        )

        learner.fit(self.X_train, self.y_train, self.X_test, self.y_test, verbose=False)

        y_pred = learner.predict(self.X_test)
        y_proba = learner.predict_proba(self.X_test)[:, 1]

        self.results['transformer_classifier'] = {
            'name': 'Transformer Classifier',
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metrics': self._compute_metrics(y_pred, y_proba),
            'n_parameters': sum(p.numel() for p in learner.model.parameters()),
            'model': learner
        }

        logger.info(f"Transformer Classifier Accuracy: {self.results['transformer_classifier']['metrics']['accuracy']:.4f}")
        return self.results['transformer_classifier']

    def evaluate_transformer_hybrid(self) -> Optional[Dict]:
        """Train and evaluate Hybrid Transformer with witness extraction."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping Hybrid Transformer")
            return None

        logger.info("Training Hybrid Transformer...")

        config = DEFAULT_TRANSFORMER_CONFIG.to_dict()
        learner = TransformerWitnessLearner(
            pauli_basis=self.basis,
            mode='hybrid',
            **config,
            random_state=self.seed
        )

        learner.fit(self.X_train, self.y_train, self.X_test, self.y_test, verbose=False)

        y_pred = learner.predict(self.X_test)
        y_proba = learner.predict_proba(self.X_test)[:, 1]

        # Get witness info
        witness = learner.get_witness_operator()
        witness_coeffs = {str(p): float(np.real(c)) for p, c in witness.to_list()}

        self.results['transformer_hybrid'] = {
            'name': 'Hybrid Transformer',
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metrics': self._compute_metrics(y_pred, y_proba),
            'n_parameters': sum(p.numel() for p in learner.model.parameters()),
            'witness_coefficients': witness_coeffs,
            'measurement_cost': learner.get_measurement_cost(),
            'model': learner
        }

        logger.info(f"Hybrid Transformer Accuracy: {self.results['transformer_hybrid']['metrics']['accuracy']:.4f}")
        return self.results['transformer_hybrid']

    def evaluate_mlp(self) -> Optional[Dict]:
        """Train and evaluate MLP Discriminator classifier."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping MLP")
            return None

        logger.info("Training MLP Discriminator...")

        config = DEFAULT_MLP_CONFIG.to_dict()
        learner = MLPClassifierLearner(
            n_features=len(self.basis),
            **config,
            random_state=self.seed
        )

        learner.fit(self.X_train, self.y_train, self.X_test, self.y_test, verbose=False)

        y_pred = learner.predict(self.X_test)
        y_proba = learner.predict_proba(self.X_test)[:, 1]

        self.results['mlp'] = {
            'name': 'MLP Discriminator',
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metrics': self._compute_metrics(y_pred, y_proba),
            'n_parameters': learner.metrics.get('n_parameters', 0),
            'model': learner
        }

        logger.info(f"MLP Test Accuracy: {self.results['mlp']['metrics']['accuracy']:.4f}")
        return self.results['mlp']

    def _compute_metrics(self, y_pred, y_proba) -> Dict:
        """Compute comprehensive ML metrics."""
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_proba)
        avg_precision = average_precision_score(self.y_test, y_proba)

        cm = confusion_matrix(self.y_test, y_pred)

        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision_curve': precision_curve.tolist(),
            'recall_curve': recall_curve.tolist(),
            'confusion_matrix': cm.tolist(),
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1]),
        }

    def run_all(self) -> Dict:
        """Run evaluation for all available models."""
        self.evaluate_svm()
        self.evaluate_mlp()
        self.evaluate_transformer_classifier()
        self.evaluate_transformer_hybrid()
        return self.results

    def get_comparison_summary(self) -> Dict:
        """Get a summary comparison of all models."""
        summary = {}
        for model_key, result in self.results.items():
            if result:
                summary[model_key] = {
                    'name': result['name'],
                    'accuracy': result['metrics']['accuracy'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall'],
                    'f1': result['metrics']['f1'],
                    'roc_auc': result['metrics']['roc_auc'],
                    'avg_precision': result['metrics']['avg_precision'],
                }
        return summary


def plot_comparative_analysis(evaluator: ModelEvaluator, save_path: Optional[Path] = None):
    """
    Generate comprehensive comparative visualization with ML-standard metrics.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    results = evaluator.results
    y_test = evaluator.y_test

    # Determine number of models
    valid_models = [(k, v) for k, v in results.items() if v is not None]
    n_models = len(valid_models)

    if n_models == 0:
        logger.error("No models to plot")
        return

    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    colors = {'svm': '#3498db', 'mlp': '#9b59b6', 'transformer_classifier': '#2ecc71', 'transformer_hybrid': '#e74c3c'}
    markers = {'svm': 'o', 'mlp': 'D', 'transformer_classifier': 's', 'transformer_hybrid': '^'}

    # =========================================================================
    # 1. ROC Curves (top-left)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    for model_key, result in valid_models:
        metrics = result['metrics']
        ax1.plot(metrics['fpr'], metrics['tpr'],
                color=colors.get(model_key, 'gray'),
                linewidth=2,
                label=f"{result['name']} (AUC={metrics['roc_auc']:.3f})")

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # =========================================================================
    # 2. Precision-Recall Curves (top-middle)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    baseline = np.sum(y_test) / len(y_test)  # Class imbalance baseline

    for model_key, result in valid_models:
        metrics = result['metrics']
        ax2.plot(metrics['recall_curve'], metrics['precision_curve'],
                color=colors.get(model_key, 'gray'),
                linewidth=2,
                label=f"{result['name']} (AP={metrics['avg_precision']:.3f})")

    ax2.axhline(baseline, color='gray', linestyle='--', alpha=0.5,
               label=f'Baseline ({baseline:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(0, 1.05)

    # =========================================================================
    # 3. Metrics Comparison Bar Chart (top-right)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])

    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for i, (model_key, result) in enumerate(valid_models):
        values = [result['metrics'][m] for m in metrics_to_plot]
        offset = (i - len(valid_models)/2 + 0.5) * width
        bars = ax3.bar(x + offset, values, width,
                      label=result['name'],
                      color=colors.get(model_key, 'gray'),
                      alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_labels, rotation=15, ha='right')
    ax3.set_ylabel('Score')
    ax3.set_title('Classification Metrics Comparison')
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1.15)
    ax3.grid(axis='y', alpha=0.3)

    # =========================================================================
    # 4-6. Confusion Matrices (bottom row)
    # =========================================================================
    for i, (model_key, result) in enumerate(valid_models[:3]):  # Max 3
        ax = fig.add_subplot(gs[1, i])

        cm = np.array(result['metrics']['confusion_matrix'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)

        # Add text annotations
        for row in range(2):
            for col in range(2):
                val = cm[row, col]
                pct = cm_normalized[row, col]
                text_color = 'white' if pct > 0.5 else 'black'
                ax.text(col, row, f'{val}\n({pct:.1%})',
                       ha='center', va='center', color=text_color, fontsize=11)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Distill.', 'Distillable'])
        ax.set_yticklabels(['Non-Distill.', 'Distillable'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{result["name"]}\nConfusion Matrix')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Comparative Analysis: SVM vs Transformer Models\n'
                'Binary Classification for 3-Qubit Distillability',
                fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    plt.show()


def plot_model_details(evaluator: ModelEvaluator, save_path: Optional[Path] = None):
    """
    Plot model-specific details: SVM support vectors, Transformer coefficients.
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib required for plotting")
        return

    results = evaluator.results

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # =========================================================================
    # 1. SVM: Feature importance from coefficients
    # =========================================================================
    ax1 = axes[0]

    if 'svm' in results and results['svm']:
        svm_model = results['svm']['model']
        if hasattr(svm_model, 'coef_'):
            coeffs = np.abs(svm_model.coef_[0])
            top_indices = np.argsort(coeffs)[-15:][::-1]
            top_values = coeffs[top_indices]

            # Get Pauli labels
            pauli_labels = [str(evaluator.basis[i]) for i in top_indices]

            ax1.barh(range(len(top_indices)), top_values, color='#3498db', alpha=0.8)
            ax1.set_yticks(range(len(top_indices)))
            ax1.set_yticklabels(pauli_labels, fontsize=8, fontfamily='monospace')
            ax1.set_xlabel('|Coefficient|')
            ax1.set_title(f'SVM: Top 15 Features\n({results["svm"]["n_support_vectors"]} support vectors)')
            ax1.invert_yaxis()
            ax1.grid(axis='x', alpha=0.3)

    # =========================================================================
    # 2. Hybrid Transformer: Witness coefficients
    # =========================================================================
    ax2 = axes[1]

    if 'transformer_hybrid' in results and results['transformer_hybrid']:
        witness_coeffs = results['transformer_hybrid'].get('witness_coefficients', {})
        if witness_coeffs:
            sorted_coeffs = sorted(witness_coeffs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            paulis = [p for p, _ in sorted_coeffs]
            values = [abs(c) for _, c in sorted_coeffs]
            signs = ['#2ecc71' if c >= 0 else '#e74c3c' for _, c in sorted_coeffs]

            ax2.barh(range(len(paulis)), values, color=signs, alpha=0.8)
            ax2.set_yticks(range(len(paulis)))
            ax2.set_yticklabels(paulis, fontsize=8, fontfamily='monospace')
            ax2.set_xlabel('|Coefficient|')
            ax2.set_title(f'Hybrid Transformer: Top 15 Witness Terms\n'
                         f'(Meas. cost: {results["transformer_hybrid"]["measurement_cost"]})')
            ax2.invert_yaxis()
            ax2.grid(axis='x', alpha=0.3)

    # =========================================================================
    # 3. Model complexity comparison
    # =========================================================================
    ax3 = axes[2]

    model_names = []
    complexities = []
    accuracies = []
    colors_list = []

    color_map = {'svm': '#3498db', 'mlp': '#9b59b6', 'transformer_classifier': '#2ecc71', 'transformer_hybrid': '#e74c3c'}

    for model_key, result in results.items():
        if result:
            model_names.append(result['name'])
            accuracies.append(result['metrics']['accuracy'])
            colors_list.append(color_map.get(model_key, 'gray'))

            if model_key == 'svm':
                complexities.append(result['n_support_vectors'])
            else:
                complexities.append(result['n_parameters'])

    scatter = ax3.scatter(complexities, accuracies, s=300, c=colors_list,
                         alpha=0.8, edgecolors='black', linewidth=2)

    for name, comp, acc in zip(model_names, complexities, accuracies):
        ax3.annotate(name, (comp, acc), xytext=(10, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax3.set_xlabel('Model Complexity\n(Support Vectors / Parameters)')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Accuracy vs Model Complexity')
    ax3.grid(alpha=0.3)

    # Use log scale if complexity varies widely
    if max(complexities) / min(complexities) > 10:
        ax3.set_xscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")

    plt.show()


def save_results(evaluator: ModelEvaluator, n_samples: int, save_path: Path):
    """Save comparison results to JSON."""
    results = {
        'n_samples': n_samples,
        'n_train': len(evaluator.X_train),
        'n_test': len(evaluator.X_test),
        'class_balance': float(np.mean(evaluator.y_test)),
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }

    for model_key, result in evaluator.results.items():
        if result:
            model_data = {
                'name': result['name'],
                'metrics': {
                    k: v for k, v in result['metrics'].items()
                    if k not in ['fpr', 'tpr', 'precision_curve', 'recall_curve']
                }
            }
            if 'n_support_vectors' in result:
                model_data['n_support_vectors'] = result['n_support_vectors']
            if 'n_parameters' in result:
                model_data['n_parameters'] = result['n_parameters']
            if 'measurement_cost' in result:
                model_data['measurement_cost'] = result['measurement_cost']
            if 'witness_coefficients' in result:
                # Only save top 20 coefficients
                sorted_coeffs = sorted(result['witness_coefficients'].items(),
                                       key=lambda x: abs(x[1]), reverse=True)[:20]
                model_data['top_witness_coefficients'] = dict(sorted_coeffs)

            results['models'][model_key] = model_data

    results['summary'] = evaluator.get_comparison_summary()

    with open(save_path, 'w') as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)

    logger.info(f"Results saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run comparative analysis of SVM vs Transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_comparative_analysis.py --n-samples 2000
  python scripts/run_comparative_analysis.py --n-samples 5000 --save-plots
  python scripts/run_comparative_analysis.py --n-samples 3000 --seed 123
        """
    )

    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of samples to generate (default: 2000)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED})')
    parser.add_argument('--noise-min', type=float, default=0.0,
                        help='Minimum noise level (default: 0.0)')
    parser.add_argument('--noise-max', type=float, default=0.5,
                        help='Maximum noise level (default: 0.5)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to figures/ directory')
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR),
                        help=f'Directory to save results (default: {RESULTS_DIR})')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    noise_range = (args.noise_min, args.noise_max)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Generate Dataset
    # =========================================================================
    logger.info("=" * 70)
    logger.info("COMPARATIVE ANALYSIS: SVM vs Transformer Models")
    logger.info("=" * 70)

    logger.info(f"\nGenerating {args.n_samples} samples...")

    states, labels = generate_distillability_dataset(
        n_samples=args.n_samples,
        noise_range=noise_range,
        seed=args.seed
    )
    labels = np.array(labels)

    # Extract features
    basis = create_sparse_measurement_set(3, 'two_body')
    features = extract_features_batch(states, basis, verbose=False)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Class balance: {np.mean(labels):.2%} distillable")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )

    # =========================================================================
    # Run Evaluation
    # =========================================================================
    evaluator = ModelEvaluator(X_train, y_train, X_test, y_test, basis, args.seed)
    evaluator.run_all()

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    summary = evaluator.get_comparison_summary()
    for model_key, metrics in summary.items():
        logger.info(f"\n{metrics['name']}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg Prec:  {metrics['avg_precision']:.4f}")

    # =========================================================================
    # Save Results
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"comparative_analysis_{timestamp}.json"
    save_results(evaluator, args.n_samples, results_path)

    # =========================================================================
    # Generate Plots
    # =========================================================================
    if MATPLOTLIB_AVAILABLE:
        logger.info("\nGenerating plots...")

        # Main comparison plot
        comparison_path = FIGURES_DIR / f"comparative_analysis_{timestamp}.png" if args.save_plots else None
        plot_comparative_analysis(evaluator, comparison_path)

        # Model details plot
        details_path = FIGURES_DIR / f"model_details_{timestamp}.png" if args.save_plots else None
        plot_model_details(evaluator, details_path)

        if args.save_plots:
            logger.info(f"\nFigures saved to: {FIGURES_DIR}")
    else:
        logger.warning("Matplotlib not available. Skipping plots.")

    logger.info("\nComparative analysis complete!")


if __name__ == '__main__':
    main()
