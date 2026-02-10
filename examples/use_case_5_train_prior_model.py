#!/usr/bin/env python3
"""
Use Case 5: Train Prior Model
Description: Train a prior HELM-GPT model on cyclic peptide sequence data
Priority: Medium
Complexity: Complex
"""

import sys
import os
import json
import argparse
import pandas as pd
import torch
from pathlib import Path

# Add the repo path to sys.path for imports
repo_path = Path(__file__).parent.parent / "repo" / "helm-gpt"
sys.path.insert(0, str(repo_path))

try:
    from model.model import GPT, GPTConfig
    from prior.trainer import Trainer, TrainerConfig
    from utils.dataset import HelmDictionary, load_seqs_from_list, get_tensor_dataset
    import utils.utils as utils
except ImportError as e:
    print(f"Error importing HELM-GPT modules: {e}")
    print("This script requires the HELM-GPT library to be installed.")
    print("Please run it in the Python 3.7 environment: mamba activate ./env_py3.7")
    sys.exit(1)

def prepare_training_data(train_file, valid_file=None, helm_column='HELM'):
    """
    Prepare training and validation data

    Args:
        train_file (str): Training data CSV file
        valid_file (str): Validation data CSV file (optional)
        helm_column (str): Column containing HELM sequences

    Returns:
        tuple: (train_sequences, valid_sequences)
    """
    # Load training data
    print(f"Loading training data from {train_file}")
    train_df = pd.read_csv(train_file)

    if helm_column not in train_df.columns:
        raise ValueError(f"Column '{helm_column}' not found in {train_file}")

    train_sequences = train_df[helm_column].dropna().tolist()
    print(f"Training sequences: {len(train_sequences)}")

    # Load validation data
    valid_sequences = []
    if valid_file and valid_file != train_file:
        print(f"Loading validation data from {valid_file}")
        valid_df = pd.read_csv(valid_file)
        if helm_column in valid_df.columns:
            valid_sequences = valid_df[helm_column].dropna().tolist()
            print(f"Validation sequences: {len(valid_sequences)}")
    else:
        # Use a subset of training data for validation
        split_idx = int(0.9 * len(train_sequences))
        valid_sequences = train_sequences[split_idx:]
        train_sequences = train_sequences[:split_idx]
        print(f"Split data - Training: {len(train_sequences)}, Validation: {len(valid_sequences)}")

    return train_sequences, valid_sequences

def main():
    parser = argparse.ArgumentParser(description="Train prior HELM-GPT model")
    parser.add_argument("--train_data",
                       default="examples/data/sequences/biotherapeutics_dict_prot_flt.csv",
                       help="Training data CSV file")
    parser.add_argument("--valid_data",
                       help="Validation data CSV file (optional)")
    parser.add_argument("--output_dir", "-o",
                       default="./trained_model",
                       help="Output directory for model checkpoints")
    parser.add_argument("--helm_column",
                       default="HELM",
                       help="Column name containing HELM sequences")

    # Model architecture parameters
    parser.add_argument("--n_layers", type=int, default=8,
                       help="Number of transformer layers (default: 8)")
    parser.add_argument("--n_embd", type=int, default=512,
                       help="Embedding dimension (default: 512)")
    parser.add_argument("--n_head", type=int, default=8,
                       help="Number of attention heads (default: 8)")
    parser.add_argument("--max_len", type=int, default=200,
                       help="Maximum sequence length (default: 200)")

    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--device", default="auto",
                       help="Device for training (cpu/cuda/auto)")

    args = parser.parse_args()

    # Check input files
    if not os.path.exists(args.train_data):
        print(f"Error: Training data file not found: {args.train_data}")
        return 1

    if args.valid_data and not os.path.exists(args.valid_data):
        print(f"Error: Validation data file not found: {args.valid_data}")
        return 1

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Prepare data
        train_sequences, valid_sequences = prepare_training_data(
            args.train_data,
            args.valid_data,
            args.helm_column
        )

        # Initialize HELM dictionary
        helm_dict = HelmDictionary()
        vocab_size = helm_dict.get_char_num()

        print(f"HELM dictionary size: {vocab_size}")

        # Create model configuration
        # block_size must be >= max_len + 1 to accommodate BEGIN/END tokens
        model_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.max_len + 1,
            n_layer=args.n_layers,
            n_head=args.n_head,
            n_embd=args.n_embd
        )

        # Create model
        model = GPT(model_config)
        model.to(device)

        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Convert sequences to datasets
        print("Preparing training datasets...")
        train_array, _ = load_seqs_from_list(train_sequences, rm_duplicates=True, max_len=args.max_len)
        train_dataset = get_tensor_dataset(train_array)

        valid_dataset = None
        if valid_sequences:
            valid_array, _ = load_seqs_from_list(valid_sequences, rm_duplicates=True, max_len=args.max_len)
            valid_dataset = get_tensor_dataset(valid_array)

        print(f"Training dataset: {len(train_dataset)} sequences")
        if valid_dataset:
            print(f"Validation dataset: {len(valid_dataset)} sequences")

        # Create training configuration
        trainer_config = TrainerConfig(
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            warmup_tokens=len(train_sequences) * args.max_len * 0.1,
            final_tokens=len(train_sequences) * args.max_len * args.n_epochs,
        )

        # Create trainer
        trainer = Trainer(model, trainer_config)

        # Save configuration
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config.__dict__, f, indent=2, default=str)

        # Save command line arguments
        args_path = os.path.join(args.output_dir, "commandline_args.json")
        with open(args_path, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

        print(f"Configuration saved to {config_path}")
        print(f"Arguments saved to {args_path}")

        # Train model
        print("Starting training...")
        trainer.fit(train_dataset, valid_dataset,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    save_model=True)

        print(f"Training completed. Model saved to {args.output_dir}")

        # Note: The trainer.fit() with save_model=True already saves
        # gpt_model_final_<loss>.pt and gpt_model_final_<loss>.json
        # via save_gpt_model(), which is the format load_gpt_model() expects.
        # Look for files matching gpt_model_final_*.pt in output_dir.
        final_files = list(Path(args.output_dir).glob("gpt_model_final_*.pt"))
        if final_files:
            print(f"Final model saved as {final_files[0]}")
        else:
            print("Warning: Final model not found. Training may have had issues.")

        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())