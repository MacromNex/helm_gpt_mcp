#!/usr/bin/env python3
"""
Use Case 1: HELM Notation to SMILES Conversion
Description: Convert cyclic peptide sequences from HELM notation to SMILES representation
Priority: High
Complexity: Simple
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# Add the repo path to sys.path for imports
repo_path = Path(__file__).parent.parent / "repo" / "helm-gpt"
sys.path.insert(0, str(repo_path))

try:
    from utils.helm_utils import get_cycpep_smi_from_helm, is_helm_valid
    from utils.dataset import HelmDictionary
except ImportError as e:
    print(f"Error importing HELM-GPT modules: {e}")
    print("This script requires the HELM-GPT library to be installed.")
    print("Please run it in the Python 3.7 environment: mamba activate ./env_py3.7")
    sys.exit(1)

def convert_helm_to_smiles(helm_sequence):
    """
    Convert a single HELM sequence to SMILES

    Args:
        helm_sequence (str): HELM notation sequence

    Returns:
        tuple: (smiles, success, error_message)
    """
    try:
        # Validate HELM sequence
        if not is_helm_valid(helm_sequence):
            return None, False, "Invalid HELM sequence"

        # Convert to SMILES
        smiles = get_cycpep_smi_from_helm(helm_sequence)
        if smiles is None:
            return None, False, "Failed to convert HELM to SMILES"

        return smiles, True, None
    except Exception as e:
        return None, False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Convert HELM notation to SMILES representation")
    parser.add_argument("--input", "-i",
                       default="examples/data/sequences/CycPeptMPDB_Peptide_All.csv",
                       help="Input CSV file with HELM sequences or single HELM string")
    parser.add_argument("--output", "-o",
                       default="helm_to_smiles_output.csv",
                       help="Output CSV file")
    parser.add_argument("--helm_column",
                       default="HELM",
                       help="Column name containing HELM sequences (default: HELM)")
    parser.add_argument("--limit", "-l", type=int, default=100,
                       help="Limit number of sequences to process (default: 100)")

    args = parser.parse_args()

    # Check if input is a file or a single HELM string
    if os.path.exists(args.input):
        print(f"Reading HELM sequences from {args.input}")

        try:
            # Read CSV file
            df = pd.read_csv(args.input)

            if args.helm_column not in df.columns:
                print(f"Error: Column '{args.helm_column}' not found in {args.input}")
                print(f"Available columns: {list(df.columns)}")
                return 1

            # Limit number of sequences
            helm_sequences = df[args.helm_column].dropna().head(args.limit).tolist()
            print(f"Processing {len(helm_sequences)} HELM sequences...")

        except Exception as e:
            print(f"Error reading input file: {e}")
            return 1
    else:
        # Treat input as a single HELM string
        helm_sequences = [args.input]
        print(f"Processing single HELM sequence: {args.input}")

    # Convert sequences
    results = []
    success_count = 0

    for i, helm_seq in enumerate(helm_sequences):
        if pd.isna(helm_seq) or helm_seq.strip() == "":
            continue

        smiles, success, error = convert_helm_to_smiles(helm_seq.strip())

        result = {
            'helm_sequence': helm_seq,
            'smiles': smiles if success else None,
            'success': success,
            'error': error if not success else None
        }
        results.append(result)

        if success:
            success_count += 1
            print(f"✓ [{i+1}/{len(helm_sequences)}] Converted successfully")
        else:
            print(f"✗ [{i+1}/{len(helm_sequences)}] Failed: {error}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    print(f"\nResults saved to {args.output}")
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    # Show sample results
    if success_count > 0:
        print("\nSample successful conversions:")
        successful_results = results_df[results_df['success'] == True].head(3)
        for _, row in successful_results.iterrows():
            print(f"HELM: {row['helm_sequence'][:80]}...")
            print(f"SMILES: {row['smiles'][:80]}...")
            print()

    return 0

if __name__ == "__main__":
    sys.exit(main())