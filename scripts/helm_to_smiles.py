#!/usr/bin/env python3
"""
Script: helm_to_smiles.py
Description: Convert cyclic peptide sequences from HELM notation to SMILES representation

Original Use Case: examples/use_case_1_helm_to_smiles.py
Dependencies Removed: Direct repo imports (now using lazy-loaded wrappers)

Usage:
    python scripts/helm_to_smiles.py --input <input_file> --output <output_file>

Example:
    python scripts/helm_to_smiles.py --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv --output results/helm_to_smiles.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import sys

# Essential packages
import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from lib.helm_utils_repo import get_cycpep_smi_from_helm, is_helm_valid, check_repo_availability

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "input_format": "csv",
    "output_format": "csv",
    "helm_column": "HELM",
    "limit": 100,
    "validate_output": True,
    "include_errors": True
}

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def convert_helm_to_smiles_single(helm_sequence: str) -> Dict[str, Any]:
    """
    Convert a single HELM sequence to SMILES

    Args:
        helm_sequence: HELM notation sequence

    Returns:
        Dict containing:
            - smiles: SMILES string or None
            - success: Boolean success status
            - error: Error message if failed
    """
    try:
        if not helm_sequence or pd.isna(helm_sequence) or helm_sequence.strip() == "":
            return {
                "smiles": None,
                "success": False,
                "error": "Empty or invalid HELM sequence"
            }

        helm_sequence = helm_sequence.strip()

        # Validate HELM sequence
        if not is_helm_valid(helm_sequence):
            return {
                "smiles": None,
                "success": False,
                "error": "Invalid HELM sequence"
            }

        # Convert to SMILES
        smiles = get_cycpep_smi_from_helm(helm_sequence)
        if smiles is None:
            return {
                "smiles": None,
                "success": False,
                "error": "Failed to convert HELM to SMILES"
            }

        return {
            "smiles": smiles,
            "success": True,
            "error": None
        }

    except Exception as e:
        return {
            "smiles": None,
            "success": False,
            "error": str(e)
        }


def run_helm_to_smiles(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convert HELM sequences to SMILES format.

    Args:
        input_file: Path to input file (CSV with HELM sequences) or single HELM string
        output_file: Path to save output CSV (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: List of conversion results
            - success_count: Number of successful conversions
            - total_count: Total number of sequences processed
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_helm_to_smiles("input.csv", "output.csv")
        >>> print(f"Success rate: {result['success_count']}/{result['total_count']}")
    """
    # Setup
    input_file = Path(input_file) if isinstance(input_file, str) else input_file
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Check repo availability
    repo_available = check_repo_availability()
    if not repo_available:
        print("Warning: HELM-GP repository functions not available. Using simplified fallback.")

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

    # Convert sequences
    results = []
    success_count = 0

    for i, helm_seq in enumerate(helm_sequences):
        if pd.isna(helm_seq) or (isinstance(helm_seq, str) and helm_seq.strip() == ""):
            continue

        conversion_result = convert_helm_to_smiles_single(helm_seq)

        result = {
            'sequence_id': i + 1,
            'helm_sequence': helm_seq,
            'smiles': conversion_result['smiles'],
            'success': conversion_result['success'],
            'error': conversion_result['error']
        }
        results.append(result)

        if conversion_result['success']:
            success_count += 1
            print(f"✓ [{i+1}/{len(helm_sequences)}] Converted successfully")
        else:
            print(f"✗ [{i+1}/{len(helm_sequences)}] Failed: {conversion_result['error']}")

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    # Show sample results
    if success_count > 0:
        print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        print("\nSample successful conversions:")
        successful_results = [r for r in results if r['success']][:3]
        for result in successful_results:
            helm_preview = result['helm_sequence'][:80] + "..." if len(result['helm_sequence']) > 80 else result['helm_sequence']
            smiles_preview = result['smiles'][:80] + "..." if len(result['smiles']) > 80 else result['smiles']
            print(f"HELM: {helm_preview}")
            print(f"SMILES: {smiles_preview}")
            print()

    return {
        "results": results,
        "success_count": success_count,
        "total_count": len(results),
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "repo_available": repo_available
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
    parser.add_argument('--limit', '-l', type=int,
                       default=DEFAULT_CONFIG['limit'],
                       help=f'Limit number of sequences to process (default: {DEFAULT_CONFIG["limit"]})')

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

    # Run
    try:
        result = run_helm_to_smiles(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **config_overrides
        )

        if result['success_count'] > 0:
            print(f"Success: Converted {result['success_count']} sequences")
            if result['output_file']:
                print(f"Output saved to: {result['output_file']}")
        else:
            print("No sequences were successfully converted")
            sys.exit(1)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())