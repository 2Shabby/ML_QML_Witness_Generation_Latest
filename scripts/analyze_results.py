#!/usr/bin/env python3
"""
Analyze Experimental Results for 3-Qubit Distillability Hypothesis

This script loads JSON results from experiments and generates:
- Summary statistics
- Hypothesis conclusion (SUPPORTED/REFUTED/INCONCLUSIVE)
- Optional LaTeX tables for publication

Usage:
    python scripts/analyze_results.py results/
    python scripts/analyze_results.py results/ --latex
    python scripts/analyze_results.py results/all_experiments_*.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Legacy hypothesis evaluation thresholds retained for result analysis.
HYPOTHESIS_THRESHOLDS = {
    'strong_support': {
        'min_accuracy_36d': 0.85,
        'max_accuracy_gap': 0.05,
        'min_per_family_accuracy': 0.80
    },
    'support': {
        'min_accuracy_36d': 0.75,
        'max_accuracy_gap': 0.10,
        'min_per_family_accuracy': 0.65
    },
    'weak_support': {
        'min_accuracy_36d': 0.65,
        'max_accuracy_gap': 0.15
    },
    'inconclusive': {
        'min_accuracy_36d': 0.55
    }
}


def load_results(path: str) -> Dict:
    """Load results from JSON file or directory."""
    path = Path(path)

    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        # Wrap single file result in dict keyed by experiment type
        exp_type = data.get('metadata', {}).get('experiment', path.stem)
        return {exp_type: data}

    if path.is_dir():
        # Load all JSON files in directory
        all_results = {}
        for json_file in sorted(path.glob('*.json')):
            with open(json_file) as f:
                data = json.load(f)
                # Use experiment type from metadata
                exp_type = data.get('metadata', {}).get('experiment', json_file.stem)
                all_results[exp_type] = data
        return all_results

    raise ValueError(f"Path {path} is not a file or directory")


def compute_summary_statistics(results: Dict) -> Dict:
    """Compute summary statistics from all experiment results."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiments_analyzed': list(results.keys())
    }

    # Ablation study summary
    if 'ablation_study' in results or 'ablation' in results:
        ablation = results.get('ablation_study') or results.get('ablation')
        summary['ablation'] = {
            'accuracy_36d': ablation['restricted_36d']['accuracy_mean'],
            'accuracy_36d_std': ablation['restricted_36d']['accuracy_std'],
            'accuracy_63d': ablation['full_63d']['accuracy_mean'],
            'accuracy_63d_std': ablation['full_63d']['accuracy_std'],
            'accuracy_gap': ablation['statistical_comparison']['accuracy_gap'],
            'p_value': ablation['statistical_comparison']['paired_ttest_p'],
            'significant': ablation['statistical_comparison']['significant_at_0.05']
        }

    # Cross-validation summary
    if 'cross_validation' in results:
        cv = results['cross_validation']
        summary['cross_validation'] = {
            'accuracy_mean': cv['aggregate_statistics']['accuracy_mean'],
            'accuracy_std': cv['aggregate_statistics']['accuracy_std'],
            'accuracy_range': [
                cv['aggregate_statistics']['accuracy_min'],
                cv['aggregate_statistics']['accuracy_max']
            ],
            'n_seeds': len(cv['seeds'])
        }

    # Per-family summary
    if 'per_family_analysis' in results or 'per_family' in results:
        pf = results.get('per_family_analysis') or results.get('per_family')
        family_accuracies = {k: v['accuracy'] for k, v in pf['per_family'].items()}
        summary['per_family'] = {
            'accuracies': family_accuracies,
            'best_family': max(family_accuracies, key=family_accuracies.get),
            'worst_family': min(family_accuracies, key=family_accuracies.get),
            'mean_accuracy': float(np.mean(list(family_accuracies.values())))
        }

    # Noise robustness summary
    if 'noise_robustness' in results:
        nr = results['noise_robustness']
        noise_accuracies = {float(k): v['accuracy'] for k, v in nr['per_noise_level'].items()}
        summary['noise_robustness'] = {
            'accuracies_by_noise': noise_accuracies,
            'accuracy_at_0': noise_accuracies.get(0.0),
            'accuracy_at_0.5': noise_accuracies.get(0.5)
        }

    # Witness coefficient summary
    if 'witness_coefficients' in results:
        wc = results['witness_coefficients']
        summary['witness'] = {
            'n_terms': wc['witness_statistics']['n_terms'],
            'n_one_body': wc['witness_statistics']['n_one_body'],
            'n_two_body': wc['witness_statistics']['n_two_body'],
            'top_5_terms': [c['pauli'] for c in wc['ranked_coefficients'][:5]],
            'one_body_importance': wc.get('importance_by_type', {}).get('one_body_fraction'),
            'two_body_importance': wc.get('importance_by_type', {}).get('two_body_fraction')
        }

    return summary


def evaluate_hypothesis(summary: Dict) -> Dict:
    """
    Determine if the research hypothesis is SUPPORTED, REFUTED, or INCONCLUSIVE.

    Hypothesis: 36D restricted (1+2 body Pauli) features can reliably distinguish
    distillable from non-distillable 3-qubit states.

    Criteria:
    - SUPPORTED: 36D accuracy >= 85%, gap with 63D < 5%
    - WEAKLY SUPPORTED: 36D accuracy >= 75%, gap < 10%
    - INCONCLUSIVE: 36D accuracy 55-75%
    - REFUTED: 36D accuracy < 55% (no better than random)
    """
    evaluation = {
        'conclusion': None,
        'confidence': None,
        'evidence': [],
        'recommendations': []
    }

    # Get key metrics
    accuracy_36d = None
    accuracy_gap = None
    p_value = None

    if 'ablation' in summary:
        accuracy_36d = summary['ablation']['accuracy_36d']
        accuracy_gap = summary['ablation']['accuracy_gap']
        p_value = summary['ablation']['p_value']
    elif 'cross_validation' in summary:
        accuracy_36d = summary['cross_validation']['accuracy_mean']

    if accuracy_36d is None:
        evaluation['conclusion'] = 'INSUFFICIENT_DATA'
        evaluation['evidence'].append('No ablation or cross-validation results found')
        return evaluation

    # Evaluate against thresholds
    thresholds = HYPOTHESIS_THRESHOLDS

    # Check for strong support
    if accuracy_36d >= thresholds['strong_support']['min_accuracy_36d']:
        if accuracy_gap is not None and accuracy_gap <= thresholds['strong_support']['max_accuracy_gap']:
            evaluation['conclusion'] = 'STRONGLY_SUPPORTED'
            evaluation['confidence'] = 'high'
            evaluation['evidence'].append(
                f"36D accuracy ({accuracy_36d:.1%}) exceeds 85% threshold"
            )
            evaluation['evidence'].append(
                f"Accuracy gap ({accuracy_gap:.1%}) within 5% of full features"
            )
        else:
            evaluation['conclusion'] = 'SUPPORTED'
            evaluation['confidence'] = 'medium-high'
            evaluation['evidence'].append(
                f"36D accuracy ({accuracy_36d:.1%}) exceeds 85% threshold"
            )
            if accuracy_gap:
                evaluation['evidence'].append(
                    f"Accuracy gap ({accuracy_gap:.1%}) with 63D features"
                )

    # Check for moderate support
    elif accuracy_36d >= thresholds['support']['min_accuracy_36d']:
        evaluation['conclusion'] = 'SUPPORTED'
        evaluation['confidence'] = 'medium'
        evaluation['evidence'].append(
            f"36D accuracy ({accuracy_36d:.1%}) between 75-85%"
        )
        if accuracy_gap:
            evaluation['evidence'].append(
                f"Gap of {accuracy_gap:.1%} with full 63D features"
            )
        evaluation['recommendations'].append(
            "Consider increasing dataset size for more robust conclusions"
        )

    # Check for weak support
    elif accuracy_36d >= thresholds['weak_support']['min_accuracy_36d']:
        evaluation['conclusion'] = 'WEAKLY_SUPPORTED'
        evaluation['confidence'] = 'low'
        evaluation['evidence'].append(
            f"36D accuracy ({accuracy_36d:.1%}) between 65-75%"
        )
        evaluation['recommendations'].append(
            "Explore nonlinear models (MLP) to check if linear boundary is limiting"
        )
        evaluation['recommendations'].append(
            "Consider adding more boundary states to dataset"
        )

    # Check for inconclusive
    elif accuracy_36d >= thresholds['inconclusive']['min_accuracy_36d']:
        evaluation['conclusion'] = 'INCONCLUSIVE'
        evaluation['confidence'] = 'very_low'
        evaluation['evidence'].append(
            f"36D accuracy ({accuracy_36d:.1%}) only marginally better than random (55-65%)"
        )
        evaluation['recommendations'].append(
            "Try L1-regularized sparse SVM for feature selection"
        )
        evaluation['recommendations'].append(
            "Investigate if 3-body correlations are essential"
        )

    # Refuted
    else:
        evaluation['conclusion'] = 'REFUTED'
        evaluation['confidence'] = 'high'
        evaluation['evidence'].append(
            f"36D accuracy ({accuracy_36d:.1%}) no better than random guessing"
        )
        evaluation['evidence'].append(
            "Restricted 1+2 body features insufficient for distillability classification"
        )

    # Add statistical significance info
    if p_value is not None:
        if p_value < 0.01:
            evaluation['evidence'].append(
                f"63D vs 36D difference highly significant (p={p_value:.4f})"
            )
        elif p_value < 0.05:
            evaluation['evidence'].append(
                f"63D vs 36D difference significant (p={p_value:.4f})"
            )
        else:
            evaluation['evidence'].append(
                f"63D vs 36D difference not significant (p={p_value:.4f})"
            )

    # Add per-family insights
    if 'per_family' in summary:
        worst = summary['per_family']['worst_family']
        worst_acc = summary['per_family']['accuracies'][worst]
        if worst_acc < 0.6:
            evaluation['recommendations'].append(
                f"Investigate poor performance on {worst} states ({worst_acc:.1%})"
            )

    return evaluation


def generate_latex_table(summary: Dict, evaluation: Dict) -> str:
    """Generate LaTeX tables for publication."""
    latex = []

    # Main results table
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{3-Qubit Distillability Classification Results}")
    latex.append(r"\label{tab:results}")
    latex.append(r"\begin{tabular}{lcc}")
    latex.append(r"\toprule")
    latex.append(r"Metric & 36D Restricted & 63D Full \\")
    latex.append(r"\midrule")

    if 'ablation' in summary:
        abl = summary['ablation']
        latex.append(
            f"Accuracy & ${abl['accuracy_36d']:.1%} \\pm {abl['accuracy_36d_std']:.1%}$ & "
            f"${abl['accuracy_63d']:.1%} \\pm {abl['accuracy_63d_std']:.1%}$ \\\\"
        )
        latex.append(f"Accuracy Gap & \\multicolumn{{2}}{{c}}{{{abl['accuracy_gap']:.1%}}} \\\\")
        latex.append(f"$p$-value & \\multicolumn{{2}}{{c}}{{{abl['p_value']:.4f}}} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")

    # Per-family table
    if 'per_family' in summary:
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Per-Family Classification Accuracy}")
        latex.append(r"\label{tab:per_family}")
        latex.append(r"\begin{tabular}{lc}")
        latex.append(r"\toprule")
        latex.append(r"State Family & Accuracy \\")
        latex.append(r"\midrule")

        for family, acc in summary['per_family']['accuracies'].items():
            latex.append(f"{family.upper()} & {acc:.1%} \\\\")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        latex.append("")

    # Top witness terms table
    if 'witness' in summary and summary['witness'].get('top_5_terms'):
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Most Important Pauli Observables}")
        latex.append(r"\label{tab:witness}")
        latex.append(r"\begin{tabular}{cl}")
        latex.append(r"\toprule")
        latex.append(r"Rank & Pauli Observable \\")
        latex.append(r"\midrule")

        for i, term in enumerate(summary['witness']['top_5_terms']):
            latex.append(f"{i+1} & $\\hat{{{term}}}$ \\\\")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

    return "\n".join(latex)


def print_summary_report(summary: Dict, evaluation: Dict):
    """Print a human-readable summary report."""
    print("\n" + "="*70)
    print("3-QUBIT DISTILLABILITY HYPOTHESIS: EXPERIMENTAL RESULTS SUMMARY")
    print("="*70)

    print(f"\nAnalysis timestamp: {summary['timestamp']}")
    print(f"Experiments analyzed: {', '.join(summary['experiments_analyzed'])}")

    # Hypothesis evaluation
    print("\n" + "-"*70)
    print("HYPOTHESIS EVALUATION")
    print("-"*70)
    print(f"\nConclusion: {evaluation['conclusion']}")
    print(f"Confidence: {evaluation['confidence']}")

    print("\nEvidence:")
    for ev in evaluation['evidence']:
        print(f"  - {ev}")

    if evaluation['recommendations']:
        print("\nRecommendations:")
        for rec in evaluation['recommendations']:
            print(f"  - {rec}")

    # Ablation study results
    if 'ablation' in summary:
        print("\n" + "-"*70)
        print("ABLATION STUDY: 36D Restricted vs 63D Full")
        print("-"*70)
        abl = summary['ablation']
        print(f"  36D Accuracy: {abl['accuracy_36d']:.1%} +/- {abl['accuracy_36d_std']:.1%}")
        print(f"  63D Accuracy: {abl['accuracy_63d']:.1%} +/- {abl['accuracy_63d_std']:.1%}")
        print(f"  Gap: {abl['accuracy_gap']:.1%}")
        print(f"  p-value: {abl['p_value']:.4f} ({'significant' if abl['significant'] else 'not significant'})")

    # Cross-validation results
    if 'cross_validation' in summary:
        print("\n" + "-"*70)
        print("CROSS-VALIDATION (Multi-seed)")
        print("-"*70)
        cv = summary['cross_validation']
        print(f"  Mean Accuracy: {cv['accuracy_mean']:.1%} +/- {cv['accuracy_std']:.1%}")
        print(f"  Range: [{cv['accuracy_range'][0]:.1%}, {cv['accuracy_range'][1]:.1%}]")
        print(f"  Seeds tested: {cv['n_seeds']}")

    # Per-family results
    if 'per_family' in summary:
        print("\n" + "-"*70)
        print("PER-FAMILY ANALYSIS")
        print("-"*70)
        pf = summary['per_family']
        for family, acc in pf['accuracies'].items():
            marker = " *" if family == pf['best_family'] else (" !" if family == pf['worst_family'] else "")
            print(f"  {family.upper():8s}: {acc:.1%}{marker}")
        print(f"\n  (* = best, ! = worst)")

    # Witness analysis
    if 'witness' in summary:
        print("\n" + "-"*70)
        print("WITNESS COEFFICIENT ANALYSIS")
        print("-"*70)
        wc = summary['witness']
        print(f"  Total terms: {wc['n_terms']}")
        print(f"  One-body terms: {wc['n_one_body']}")
        print(f"  Two-body terms: {wc['n_two_body']}")
        if wc.get('one_body_importance'):
            print(f"  One-body importance: {wc['one_body_importance']:.1%}")
            print(f"  Two-body importance: {wc['two_body_importance']:.1%}")
        print(f"  Top 5 terms: {', '.join(wc['top_5_terms'])}")

    # Noise robustness
    if 'noise_robustness' in summary:
        print("\n" + "-"*70)
        print("NOISE ROBUSTNESS")
        print("-"*70)
        nr = summary['noise_robustness']
        for noise, acc in sorted(nr['accuracies_by_noise'].items()):
            print(f"  Noise={noise:.2f}: {acc:.1%}")

    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze experimental results for 3-qubit distillability hypothesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_results.py results/
  python scripts/analyze_results.py results/all_experiments_*.json
  python scripts/analyze_results.py results/ --latex --output results/summary.tex
        """
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to results JSON file or directory'
    )

    parser.add_argument(
        '--latex',
        action='store_true',
        help='Generate LaTeX tables'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for LaTeX tables'
    )

    parser.add_argument(
        '--json-output',
        type=str,
        default=None,
        help='Save summary as JSON'
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.path}...")
    results = load_results(args.path)

    # Compute summary
    print("Computing summary statistics...")
    summary = compute_summary_statistics(results)

    # Evaluate hypothesis
    print("Evaluating hypothesis...")
    evaluation = evaluate_hypothesis(summary)

    # Print report
    print_summary_report(summary, evaluation)

    # Generate LaTeX if requested
    if args.latex:
        latex = generate_latex_table(summary, evaluation)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(latex)
            print(f"LaTeX tables saved to {args.output}")
        else:
            print("\n" + "-"*70)
            print("LATEX TABLES")
            print("-"*70)
            print(latex)

    # Save summary JSON if requested
    if args.json_output:
        output_data = {
            'summary': summary,
            'evaluation': evaluation
        }
        with open(args.json_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Summary saved to {args.json_output}")

    # Return evaluation conclusion for scripting
    return evaluation['conclusion']


if __name__ == '__main__':
    conclusion = main()
    # Exit with code based on conclusion
    exit_codes = {
        'STRONGLY_SUPPORTED': 0,
        'SUPPORTED': 0,
        'WEAKLY_SUPPORTED': 0,
        'INCONCLUSIVE': 1,
        'REFUTED': 2,
        'INSUFFICIENT_DATA': 3
    }
    sys.exit(exit_codes.get(conclusion, 1))
