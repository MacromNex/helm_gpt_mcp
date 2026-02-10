#!/usr/bin/env python3
"""
Script: predict_kras_binding.py
Description: Predict KRAS protein binding affinity (KD) for cyclic peptides

Original Use Case: examples/use_case_4_predict_kras_binding.py
Dependencies Removed: Direct repo imports (now using lazy-loaded wrappers)

Usage:
    python scripts/predict_kras_binding.py --input <input_file> --output <output_file>

Example:
    python scripts/predict_kras_binding.py --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv --output results/kras_binding.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import sys

# Essential scientific packages
import pandas as pd
import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from lib.prediction_utils import KRASBindingPredictor, check_prediction_models

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "input_format": "csv",
    "output_format": "csv",
    "helm_column": "HELM",
    "limit": 100,
    "model_path": None,  # Will use default if None
    "include_statistics": True,
    "show_top_binders": 5
}

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_kras_binding(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict KRAS protein binding affinity for cyclic peptides.

    Args:
        input_file: Path to input file (CSV with HELM sequences) or single HELM string
        output_file: Path to save output CSV (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: List of prediction results
            - success_count: Number of successful predictions
            - total_count: Total number of sequences processed
            - statistics: Statistical summary of results
            - top_binders: Best binding sequences
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_predict_kras_binding("input.csv", "output.csv")
        >>> print(f"Mean KD: {result['statistics']['mean_kd']:.2f} μM")
    """
    # Setup
    input_file = Path(input_file) if isinstance(input_file, str) else input_file
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Check model availability
    model_status = check_prediction_models()
    print(f"Repo available: {model_status['repo_available']}")
    print(f"KRAS model available: {model_status['kras_model']}")

    # Initialize predictor
    model_path = config.get("model_path")
    predictor = KRASBindingPredictor(model_path=model_path)

    # Load input data
    if input_file.exists():
        print(f"Reading HELM sequences from {input_file}")
        try:
            df = pd.read_csv(input_file)
            helm_column = config.get("helm_column", "HELM")

            if helm_column not in df.columns:
                available_cols = list(df.columns)
                raise ValueError(f"Column '{helm_column}' not found. Available columns: {available_cols}")

            # Limit number of sequences
            limit = config.get("limit", 100)
            helm_sequences = df[helm_column].dropna().head(limit).tolist()
            print(f"Processing {len(helm_sequences)} HELM sequences...")

        except Exception as e:
            raise ValueError(f"Error reading input file: {e}")

    else:
        # Treat input as a single HELM string
        helm_sequences = [str(input_file)]
        print(f"Processing single HELM sequence")

    # Predict KRAS binding
    print("Predicting KRAS binding affinity...")
    try:
        scores, kd_values = predictor.predict_batch(helm_sequences)
    except Exception as e:
        print(f"Error during prediction: {e}")
        scores = [None] * len(helm_sequences)
        kd_values = [None] * len(helm_sequences)

    # Prepare results
    results = []
    success_count = 0

    for i, (helm_seq, score, kd_value) in enumerate(zip(helm_sequences, scores, kd_values)):
        if score is not None:
            success_count += 1

        # Interpret results
        binding_interpretation = predictor.interpret_binding(kd_value)
        score_interpretation = predictor.interpret_score(score)

        result = {
            'sequence_id': i + 1,
            'helm_sequence': helm_seq,
            'binding_score': score,
            'kd_uM': kd_value,
            'kd_interpretation': binding_interpretation,
            'score_interpretation': score_interpretation,
            'prediction_success': score is not None
        }
        results.append(result)

        if score is not None:
            print(f"✓ [{i+1}/{len(helm_sequences)}] Score: {score:.3f}, KD: {kd_value:.2f} μM ({binding_interpretation})")
        else:
            print(f"✗ [{i+1}/{len(helm_sequences)}] Prediction failed")

    # Calculate statistics
    statistics = {}
    if success_count > 0:
        successful_scores = [r['binding_score'] for r in results if r['binding_score'] is not None]
        successful_kd = [r['kd_uM'] for r in results if r['kd_uM'] is not None]

        statistics = {
            'mean_score': np.mean(successful_scores),
            'median_score': np.median(successful_scores),
            'mean_kd': np.mean(successful_kd),
            'median_kd': np.median(successful_kd),
            'min_kd': np.min(successful_kd),
            'max_kd': np.max(successful_kd),
            'std_kd': np.std(successful_kd)
        }

        # Count by binding strength
        interpretations = [r['kd_interpretation'] for r in results if r['prediction_success']]
        interpretation_counts = {}
        for category in [
            "Strong binding (KD < 1 μM)",
            "Moderate binding (1-10 μM)",
            "Weak binding (10-100 μM)",
            "Very weak/no binding (KD > 100 μM)"
        ]:
            count = interpretations.count(category)
            interpretation_counts[category] = {
                'count': count,
                'percentage': count / success_count * 100 if success_count > 0 else 0
            }
        statistics['interpretation_counts'] = interpretation_counts

    # Find top binders
    top_binders = []
    if success_count > 0:
        successful_results = [r for r in results if r['prediction_success']]
        # Sort by KD (lower is better)
        successful_results.sort(key=lambda x: x['kd_uM'] if x['kd_uM'] is not None else float('inf'))
        top_count = min(config.get("show_top_binders", 5), len(successful_results))
        top_binders = successful_results[:top_count]

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    # Display summary
    print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    if config.get("include_statistics", True) and statistics:
        print(f"\nKRAS Binding Statistics:")
        print(f"Mean binding score: {statistics['mean_score']:.3f}")
        print(f"Mean KD: {statistics['mean_kd']:.2f} μM")
        print(f"Median KD: {statistics['median_kd']:.2f} μM")
        print(f"KD range: {statistics['min_kd']:.2f} - {statistics['max_kd']:.2f} μM")

        print(f"\nBinding strength distribution:")
        for category, counts in statistics['interpretation_counts'].items():
            if counts['count'] > 0:
                print(f"{category}: {counts['count']} ({counts['percentage']:.1f}%)")

    # Show top binders
    if top_binders:
        print(f"\nTop {len(top_binders)} KRAS binders (lowest KD):")
        for i, binder in enumerate(top_binders, 1):
            print(f"{i}. KD: {binder['kd_uM']:.2f} μM, Score: {binder['binding_score']:.3f}")

    return {
        "results": results,
        "success_count": success_count,
        "total_count": len(results),
        "statistics": statistics,
        "top_binders": top_binders,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "model_status": model_status
        }
    }


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with HELM sequences or single HELM string')
    parser.add_argument('--output', '-o',
                       help='Output CSV file path')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--helm_column',
                       default=DEFAULT_CONFIG['helm_column'],
                       help=f'Column name containing HELM sequences (default: {DEFAULT_CONFIG["helm_column"]})')
    parser.add_argument('--model',
                       help='Path to KRAS model (.pkl file)')
    parser.add_argument('--limit', '-l', type=int,
                       default=DEFAULT_CONFIG['limit'],
                       help=f'Limit number of sequences to process (default: {DEFAULT_CONFIG["limit"]})')
    parser.add_argument('--top-binders', type=int,
                       default=DEFAULT_CONFIG['show_top_binders'],
                       help=f'Number of top binders to show (default: {DEFAULT_CONFIG["show_top_binders"]})')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI arguments
    config_overrides = {}
    if args.helm_column != DEFAULT_CONFIG['helm_column']:
        config_overrides['helm_column'] = args.helm_column
    if args.limit != DEFAULT_CONFIG['limit']:
        config_overrides['limit'] = args.limit
    if args.top_binders != DEFAULT_CONFIG['show_top_binders']:
        config_overrides['show_top_binders'] = args.top_binders
    if args.model:
        config_overrides['model_path'] = args.model

    # Run
    try:
        result = run_predict_kras_binding(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **config_overrides
        )

        if result['success_count'] > 0:
            print(f"Success: Predicted KRAS binding for {result['success_count']} sequences")
            if result['output_file']:
                print(f"Output saved to: {result['output_file']}")
        else:
            print("No KRAS binding predictions were successful")
            sys.exit(1)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())