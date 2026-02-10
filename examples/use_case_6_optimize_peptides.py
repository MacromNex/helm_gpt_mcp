#!/usr/bin/env python3
"""
Use Case 6: Optimize Peptides with Agent Training
Description: Train an agent to optimize cyclic peptides for specific properties (permeability, KRAS binding)
Priority: High
Complexity: Complex
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
    from agent.agent_trainer import AgentTrainer
    from agent.scoring_functions import ScoringFunctions
    from utils.dataset import HelmDictionary
    import utils.utils as utils
except ImportError as e:
    print(f"Error importing HELM-GPT modules: {e}")
    print("This script requires the HELM-GPT library to be installed.")
    print("Please run it in the Python 3.7 environment: mamba activate ./env_py3.7")
    sys.exit(1)

def load_prior_model(model_path, device='cpu'):
    """
    Load a pre-trained prior model

    Args:
        model_path (str): Path to prior model checkpoint
        device (str): Device to load model on

    Returns:
        GPT: Loaded prior model
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)

        if 'config' in checkpoint:
            config = GPTConfig(**checkpoint['config'])
        else:
            # Default config
            config = GPTConfig(
                vocab_size=76,
                block_size=200,
                n_layer=8,
                n_head=8,
                n_embd=512
            )

        model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()

        return model

    except Exception as e:
        raise Exception(f"Failed to load prior model from {model_path}: {e}")

def setup_scoring_functions(task, model_paths=None):
    """
    Setup scoring functions for optimization

    Args:
        task (str): Optimization task ('permeability', 'kras_kd', 'kras_perm')
        model_paths (dict): Custom model paths

    Returns:
        ScoringFunctions: Configured scoring functions
    """
    scoring_config = {}

    if task == 'permeability':
        scoring_config['permeability'] = {
            'weight': 1.0,
            'model_path': model_paths.get('permeability') if model_paths else None
        }
    elif task == 'kras_kd':
        scoring_config['kras_kd'] = {
            'weight': 1.0,
            'model_path': model_paths.get('kras_kd') if model_paths else None
        }
    elif task == 'kras_perm':
        # Multi-objective: KRAS binding + permeability
        scoring_config['kras_kd'] = {
            'weight': 0.5,
            'model_path': model_paths.get('kras_kd') if model_paths else None
        }
        scoring_config['permeability'] = {
            'weight': 0.5,
            'model_path': model_paths.get('permeability') if model_paths else None
        }
    else:
        raise ValueError(f"Unknown task: {task}")

    return ScoringFunctions(scoring_config)

def main():
    parser = argparse.ArgumentParser(description="Train agent to optimize cyclic peptides")
    parser.add_argument("--prior", required=True,
                       help="Path to pre-trained prior model (.pt file)")
    parser.add_argument("--agent",
                       help="Path to existing agent model for continued training (optional)")
    parser.add_argument("--output_dir", "-o",
                       default="./optimized_agent",
                       help="Output directory for agent checkpoints")
    parser.add_argument("--task", required=True,
                       choices=['permeability', 'kras_kd', 'kras_perm'],
                       help="Optimization task")

    # Training parameters
    parser.add_argument("--n_steps", type=int, default=1000,
                       help="Number of training steps (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--sigma", type=float, default=60,
                       help="Augmented likelihood sigma (default: 60)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate (default: 1e-5)")
    parser.add_argument("--max_len", type=int, default=200,
                       help="Maximum sequence length (default: 200)")

    # Loss function parameters
    parser.add_argument("--loss_type", default="reinvent_cpl",
                       choices=['reinvent', 'cpl', 'reinvent_cpl'],
                       help="Loss function type (default: reinvent_cpl)")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Alpha parameter for CPL loss (default: 1.0)")

    # Model paths
    parser.add_argument("--permeability_model",
                       help="Path to permeability model (.pkl file)")
    parser.add_argument("--kras_model",
                       help="Path to KRAS binding model (.pkl file)")

    parser.add_argument("--device", default="auto",
                       help="Device for training (cpu/cuda/auto)")

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.prior):
        print(f"Error: Prior model not found: {args.prior}")
        return 1

    if args.agent and not os.path.exists(args.agent):
        print(f"Error: Agent model not found: {args.agent}")
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
        # Load prior model
        print(f"Loading prior model from {args.prior}")
        prior = load_prior_model(args.prior, device)
        print("Prior model loaded successfully")

        # Load or create agent model
        if args.agent:
            print(f"Loading existing agent model from {args.agent}")
            agent = load_prior_model(args.agent, device)
        else:
            print("Creating new agent model from prior")
            agent = load_prior_model(args.prior, device)

        # Set agent to training mode
        agent.train()
        print("Agent model ready for training")

        # Setup scoring functions
        model_paths = {}
        if args.permeability_model:
            model_paths['permeability'] = args.permeability_model
        if args.kras_model:
            model_paths['kras_kd'] = args.kras_model

        print(f"Setting up scoring functions for task: {args.task}")
        scoring_functions = setup_scoring_functions(args.task, model_paths)

        # Initialize HELM dictionary
        helm_dict = HelmDictionary()

        # Create agent trainer
        print("Initializing agent trainer...")
        trainer = AgentTrainer(
            prior=prior,
            agent=agent,
            scoring_functions=scoring_functions,
            helm_dict=helm_dict,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_len=args.max_len,
            device=device,
            output_dir=args.output_dir,
            loss_type=args.loss_type,
            alpha=args.alpha,
            sigma=args.sigma
        )

        # Save configuration
        config = {
            'task': args.task,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'sigma': args.sigma,
            'learning_rate': args.learning_rate,
            'loss_type': args.loss_type,
            'alpha': args.alpha,
            'max_len': args.max_len,
            'prior_model': args.prior,
            'agent_model': args.agent,
        }

        config_path = os.path.join(args.output_dir, "agent_config.json")
        utils.save_config(config, config_path)
        print(f"Configuration saved to {config_path}")

        # Train agent
        print(f"Starting agent training for {args.n_steps} steps...")
        trainer.train(n_steps=args.n_steps)

        print(f"Training completed. Agent saved to {args.output_dir}")

        # Generate sample sequences with trained agent
        print("Generating sample sequences with trained agent...")
        sample_output_path = os.path.join(args.output_dir, "sample_sequences.csv")

        # Generate sequences
        agent.eval()
        with torch.no_grad():
            sample_sequences = []
            for _ in range(100):  # Generate 100 sample sequences
                # Sample from agent
                sequence = trainer.sample_sequences(1, temperature=0.8)[0]
                sample_sequences.append(sequence)

        # Score the sample sequences
        sample_scores = scoring_functions.score(sample_sequences)

        # Save results
        sample_df = pd.DataFrame({
            'sequence': sample_sequences,
            'score': sample_scores['total_score'],
            'task': args.task
        })
        sample_df.to_csv(sample_output_path, index=False)
        print(f"Sample sequences saved to {sample_output_path}")

        # Show statistics
        print(f"\nSample sequence statistics:")
        print(f"Mean score: {sample_df['score'].mean():.3f}")
        print(f"Max score: {sample_df['score'].max():.3f}")
        print(f"Min score: {sample_df['score'].min():.3f}")

        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())