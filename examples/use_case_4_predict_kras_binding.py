#!/usr/bin/env python3
"""
Use Case 4: Predict KRAS Binding Affinity
Description: Predict KRAS protein binding affinity (KD) for cyclic peptides
Priority: High
Complexity: Medium
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add the repo path to sys.path for imports
repo_path = Path(__file__).parent.parent / "repo" / "helm-gpt"
sys.path.insert(0, str(repo_path))

try:
    from utils.helm_utils import get_cycpep_smi_from_helm, is_helm_valid
    from agent.scoring.kras import KRASInhibition
except ImportError as e:
    print(f"Error importing HELM-GPT modules: {e}")
    print("This script requires the HELM-GPT library and dependencies to be installed.")
    print("Please run it in the Python 3.7 environment: mamba activate ./env_py3.7")
    sys.exit(1)

def predict_kras_binding_batch(helm_sequences, model_path=None):
    """
    Predict KRAS binding affinity for a batch of HELM sequences

    Args:
        helm_sequences (list): List of HELM sequences
        model_path (str): Path to KRAS model (optional)

    Returns:
        list: KRAS binding scores and KD predictions
    """
    try:
        # Initialize KRAS predictor
        if model_path:
            kras_scorer = KRASInhibition(model_path=model_path)
        else:
            # Use default model path
            default_model = "examples/data/models/kras_xgboost_reg.pkl"
            if os.path.exists(default_model):
                kras_scorer = KRASInhibition(model_path=default_model)
            else:
                kras_scorer = KRASInhibition()

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

        # Predict KRAS binding for HELM sequences
        try:
            # The KRASInhibition class works directly with HELM sequences
            scores = kras_scorer.get_scores(helm_sequences)

            # Convert scores to KD values (reverse transform from the model)
            # Based on the model design: scores are log10(KD), so KD = 10^scores
            kd_values = [10**score if score is not None else None for score in scores]
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            scores = [None] * len(helm_sequences)
            kd_values = [None] * len(helm_sequences)

        return scores, kd_values

    except Exception as e:
        print(f"Error in batch prediction: {e}")
        return [None] * len(helm_sequences), [None] * len(helm_sequences)

def main():
    parser = argparse.ArgumentParser(description="Predict KRAS binding affinity for cyclic peptides")
    parser.add_argument("--input", "-i",
                       default="examples/data/sequences/CycPeptMPDB_Peptide_All.csv",
                       help="Input CSV file with HELM sequences or single HELM string")
    parser.add_argument("--output", "-o",
                       default="kras_binding_predictions.csv",
                       help="Output CSV file")
    parser.add_argument("--helm_column",
                       default="HELM",
                       help="Column name containing HELM sequences (default: HELM)")
    parser.add_argument("--model",
                       help="Path to KRAS model (.pkl file)")
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

    # Predict KRAS binding
    print("Predicting KRAS binding affinity...")
    scores, kd_values = predict_kras_binding_batch(helm_sequences, args.model)

    # Prepare results
    results = []
    success_count = 0

    for i, (helm_seq, score, kd_value) in enumerate(zip(helm_sequences, scores, kd_values)):
        if score is not None:
            success_count += 1

        result = {
            'sequence_id': i + 1,
            'helm_sequence': helm_seq,
            'binding_score': score,
            'kd_uM': kd_value,
            'prediction_success': score is not None
        }

        # Interpret results
        if score is not None and kd_value is not None:
            if kd_value < 1.0:
                binding_interpretation = "Strong binding (KD < 1 μM)"
            elif kd_value < 10.0:
                binding_interpretation = "Moderate binding (1-10 μM)"
            elif kd_value < 100.0:
                binding_interpretation = "Weak binding (10-100 μM)"
            else:
                binding_interpretation = "Very weak/no binding (KD > 100 μM)"

            if score > 0.8:
                score_interpretation = "High binding score"
            elif score > 0.5:
                score_interpretation = "Medium binding score"
            else:
                score_interpretation = "Low binding score"
        else:
            binding_interpretation = "Prediction failed"
            score_interpretation = "Prediction failed"

        result['kd_interpretation'] = binding_interpretation
        result['score_interpretation'] = score_interpretation
        results.append(result)

        if score is not None:
            print(f"✓ [{i+1}/{len(helm_sequences)}] Score: {score:.3f}, KD: {kd_value:.2f} μM ({binding_interpretation})")
        else:
            print(f"✗ [{i+1}/{len(helm_sequences)}] Prediction failed")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)

    print(f"\nResults saved to {args.output}")
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    # Show statistics
    if success_count > 0:
        successful_scores = [r['binding_score'] for r in results if r['binding_score'] is not None]
        successful_kd = [r['kd_uM'] for r in results if r['kd_uM'] is not None]

        print(f"\nKRAS Binding Statistics:")
        print(f"Mean binding score: {np.mean(successful_scores):.3f}")
        print(f"Mean KD: {np.mean(successful_kd):.2f} μM")
        print(f"Median KD: {np.median(successful_kd):.2f} μM")
        print(f"KD range: {np.min(successful_kd):.2f} - {np.max(successful_kd):.2f} μM")

        # Count by binding strength
        interpretations = [r['kd_interpretation'] for r in results if r['prediction_success']]
        for category in ["Strong binding (KD < 1 μM)", "Moderate binding (1-10 μM)",
                        "Weak binding (10-100 μM)", "Very weak/no binding (KD > 100 μM)"]:
            count = interpretations.count(category)
            if count > 0:
                print(f"{category}: {count} ({count/success_count*100:.1f}%)")

        # Show top binders
        print(f"\nTop 5 KRAS binders (lowest KD):")
        results_df_success = results_df[results_df['prediction_success'] == True].copy()
        if len(results_df_success) > 0:
            top_binders = results_df_success.nsmallest(5, 'kd_uM')
            for idx, row in top_binders.iterrows():
                print(f"KD: {row['kd_uM']:.2f} μM, Score: {row['binding_score']:.3f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())