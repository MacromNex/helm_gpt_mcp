#!/usr/bin/env python3
"""
Use Case 2: Generate Novel Cyclic Peptides
Description: Generate novel cyclic peptide sequences using a trained HELM-GPT model
Priority: High
Complexity: Medium
"""

import sys
import os
import argparse
import pandas as pd
import torch
from pathlib import Path

# Add the repo path to sys.path for imports
repo_path = Path(__file__).parent.parent / "repo" / "helm-gpt"
sys.path.insert(0, str(repo_path))

try:
    from model.model import GPT, GPTConfig
    from model.sampler import sample_sequences
    from utils.dataset import HelmDictionary
    from utils.helm_utils import get_validity
except ImportError as e:
    print(f"Error importing HELM-GPT modules: {e}")
    print("This script requires the HELM-GPT library to be installed.")
    print("Please run it in the Python 3.7 environment: mamba activate ./env_py3.7")
    sys.exit(1)

def load_model(model_path, device='cpu'):
    """
    Load a trained GPT model

    Args:
        model_path (str): Path to model checkpoint
        device (str): Device to load model on

    Returns:
        GPT: Loaded model
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Create model config
        if 'config' in checkpoint:
            config = GPTConfig(**checkpoint['config'])
        else:
            # Default config if not saved
            config = GPTConfig(
                vocab_size=76,  # HELM dictionary size
                block_size=200,
                n_layer=8,
                n_head=8,
                n_embd=512
            )

        # Create and load model
        model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()

        return model

    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {e}")

def generate_sequences(model, n_samples=1000, max_len=200, temperature=0.8, device='cpu'):
    """
    Generate peptide sequences using the model

    Args:
        model (GPT): Trained GPT model
        n_samples (int): Number of sequences to generate
        max_len (int): Maximum sequence length
        temperature (float): Sampling temperature (higher = more diverse)
        device (str): Device for computation

    Returns:
        list: Generated HELM sequences
    """
    helm_dict = HelmDictionary()

    # Generate sequences in batches
    all_sequences = []
    batch_size = 64

    print(f"Generating {n_samples} sequences in batches of {batch_size}...")

    for i in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - i)

        # Sample sequences
        with torch.no_grad():
            sequences = sample_sequences(
                model,
                n_samples=current_batch_size,
                max_len=max_len,
                temperature=temperature,
                device=device
            )

        # Decode sequences
        for seq in sequences:
            helm_seq = helm_dict.decode(seq)
            all_sequences.append(helm_seq)

        print(f"Generated {len(all_sequences)}/{n_samples} sequences")

    return all_sequences

def main():
    parser = argparse.ArgumentParser(description="Generate novel cyclic peptide sequences")
    parser.add_argument("--model", "-m",
                       help="Path to trained GPT model (.pt file)")
    parser.add_argument("--output", "-o",
                       default="generated_peptides.csv",
                       help="Output CSV file")
    parser.add_argument("--n_samples", "-n", type=int, default=1000,
                       help="Number of sequences to generate (default: 1000)")
    parser.add_argument("--max_len", type=int, default=200,
                       help="Maximum sequence length (default: 200)")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                       help="Sampling temperature (default: 0.8)")
    parser.add_argument("--device", default="cpu",
                       help="Device for computation (cpu/cuda)")

    args = parser.parse_args()

    # Check if model path is provided
    if not args.model:
        print("Error: Please provide a model path using --model")
        print("Example: python use_case_2_generate_peptides.py --model path/to/model.pt")
        return 1

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"Using device: {device}")

    try:
        # Load model
        print(f"Loading model from {args.model}")
        model = load_model(args.model, device)
        print("Model loaded successfully")

        # Generate sequences
        sequences = generate_sequences(
            model,
            n_samples=args.n_samples,
            max_len=args.max_len,
            temperature=args.temperature,
            device=device
        )

        # Validate sequences
        print("Validating generated sequences...")
        validity = get_validity(sequences)
        valid_sequences = [seq for seq, valid in zip(sequences, validity) if valid]

        print(f"Generated {len(sequences)} sequences")
        print(f"Valid sequences: {len(valid_sequences)}/{len(sequences)} ({len(valid_sequences)/len(sequences)*100:.1f}%)")

        # Create results DataFrame
        results = []
        for i, (seq, valid) in enumerate(zip(sequences, validity)):
            results.append({
                'sequence_id': i + 1,
                'helm_sequence': seq,
                'is_valid': valid,
                'length': len(seq.replace('.', '').replace('[', '').replace(']', ''))
            })

        results_df = pd.DataFrame(results)

        # Save results
        results_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

        # Show statistics
        print(f"\nStatistics:")
        print(f"Total generated: {len(sequences)}")
        print(f"Valid: {len(valid_sequences)} ({len(valid_sequences)/len(sequences)*100:.1f}%)")
        print(f"Average length: {results_df['length'].mean():.1f}")
        print(f"Min/Max length: {results_df['length'].min()}/{results_df['length'].max()}")

        # Show sample sequences
        if len(valid_sequences) > 0:
            print("\nSample valid sequences:")
            for i, seq in enumerate(valid_sequences[:3]):
                print(f"{i+1}: {seq}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())