#!/usr/bin/env python3
"""
Use Case 3: Predict Cell Permeability
Description: Predict cell membrane permeability for cyclic peptides using trained models
Priority: High
Complexity: Medium
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Add the repo path to sys.path for imports
repo_path = Path(__file__).parent.parent / "repo" / "helm-gpt"
sys.path.insert(0, str(repo_path))

try:
    from utils.helm_utils import get_cycpep_smi_from_helm, is_helm_valid
    from agent.scoring.permeability import Permeability
except ImportError as e:
    print(f"Error importing HELM-GPT modules: {e}")
    print("This script requires the HELM-GPT library and dependencies to be installed.")
    print("Please run it in the Python 3.7 environment: mamba activate ./env_py3.7")
    sys.exit(1)

def predict_permeability_batch(helm_sequences, model_path=None):
    """
    Predict permeability for a batch of HELM sequences

    Args:
        helm_sequences (list): List of HELM sequences
        model_path (str): Path to permeability model (optional)

    Returns:
        list: Permeability scores
    """
    try:
        # Initialize permeability predictor
        if model_path:
            permeability_scorer = Permeability(model_path=model_path)
        else:
            # Use default model path
            default_model = "examples/data/models/regression_rf.pkl"
            if os.path.exists(default_model):
                permeability_scorer = Permeability(model_path=default_model)
            else:
                permeability_scorer = Permeability()

        # Convert HELM to SMILES
        smiles_list = []
        valid_indices = []

        for i, helm_seq in enumerate(helm_sequences):
            try:
                if pd.isna(helm_seq) or helm_seq.strip() == "":
                    smiles_list.append(None)
                    continue

                if is_helm_valid(helm_seq):
                    smiles = get_cycpep_smi_from_helm(helm_seq)
                    smiles_list.append(smiles)
                    if smiles:
                        valid_indices.append(i)
                else:
                    smiles_list.append(None)
            except Exception as e:
                print(f"Warning: Failed to convert HELM to SMILES for sequence {i}: {e}")
                smiles_list.append(None)

        # Predict permeability for HELM sequences
        try:
            # The Permeability class works directly with HELM sequences
            scores = permeability_scorer.get_scores(helm_sequences)
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            scores = [None] * len(helm_sequences)

        return scores

    except Exception as e:
        print(f"Error in batch prediction: {e}")
        return [None] * len(helm_sequences)

def main():
    parser = argparse.ArgumentParser(description="Predict cell permeability for cyclic peptides")
    parser.add_argument("--input", "-i",
                       default="examples/data/sequences/CycPeptMPDB_Peptide_All.csv",
                       help="Input CSV file with HELM sequences or single HELM string")
    parser.add_argument("--output", "-o",
                       default="permeability_predictions.csv",
                       help="Output CSV file")
    parser.add_argument("--helm_column",
                       default="HELM",
                       help="Column name containing HELM sequences (default: HELM)")
    parser.add_argument("--model",
                       help="Path to permeability model (.pkl file)")
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

    # Predict permeability
    print("Predicting permeability...")
    scores = predict_permeability_batch(helm_sequences, args.model)

    # Prepare results
    results = []
    success_count = 0

    for i, (helm_seq, score) in enumerate(zip(helm_sequences, scores)):
        if score is not None:
            success_count += 1

        result = {
            'sequence_id': i + 1,
            'helm_sequence': helm_seq,
            'permeability_score': score,
            'prediction_success': score is not None
        }

        # Interpret score
        if score is not None:
            if score > 0.7:
                interpretation = "High permeability"
            elif score > 0.4:
                interpretation = "Medium permeability"
            else:
                interpretation = "Low permeability"
        else:
            interpretation = "Prediction failed"

        result['interpretation'] = interpretation
        results.append(result)

        if score is not None:
            print(f"✓ [{i+1}/{len(helm_sequences)}] Score: {score:.3f} ({interpretation})")
        else:
            print(f"✗ [{i+1}/{len(helm_sequences)}] Prediction failed")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    print(f"\nResults saved to {args.output}")
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    # Show statistics
    if success_count > 0:
        successful_scores = [r['permeability_score'] for r in results if r['permeability_score'] is not None]
        print(f"\nPermeability Statistics:")
        print(f"Mean score: {np.mean(successful_scores):.3f}")
        print(f"Median score: {np.median(successful_scores):.3f}")
        print(f"Min/Max: {np.min(successful_scores):.3f}/{np.max(successful_scores):.3f}")

        # Count by interpretation
        interpretations = [r['interpretation'] for r in results if r['prediction_success']]
        for category in ["High permeability", "Medium permeability", "Low permeability"]:
            count = interpretations.count(category)
            print(f"{category}: {count} ({count/success_count*100:.1f}%)")

    return 0

if __name__ == "__main__":
    sys.exit(main())