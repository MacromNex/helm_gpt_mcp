#!/usr/bin/env python3
"""
Script: optimize_peptides.py
Description: Optimize cyclic peptides for specific properties using reinforcement learning

This script uses the HELM-GPT agent training framework to optimize peptides
for permeability, KRAS binding affinity, or both simultaneously.

Usage:
    python scripts/optimize_peptides.py --prior <prior_model.pt> --task permeability --output_dir ./results

Example:
    python scripts/optimize_peptides.py --prior models/prior.pt --task kras_perm --n_steps 500 --batch_size 32
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import sys
import json
import copy
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

# Essential scientific packages
import pandas as pd
import numpy as np

# ==============================================================================
# Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    "n_steps": 500,
    "batch_size": 32,
    "sigma": 60,
    "learning_rate": 1e-4,
    "max_len": 140,
    "loss_type": "reinvent_cpl",
    "alpha": 1.0,
    "save_every_n_steps": 100,
    "device": "auto",
    "use_amp": True,
    "sample_sequences_count": 50
}

VALID_TASKS = ["permeability", "kras_kd", "kras_perm"]

# ==============================================================================
# Lazy Loading for HELM-GPT Repository
# ==============================================================================
_repo_loaded = False
_repo_modules = {}


def _load_repo_modules() -> bool:
    """Lazy load HELM-GPT repository modules."""
    global _repo_loaded, _repo_modules

    if _repo_loaded:
        return True

    try:
        # Add repo to path (use resolve() to get absolute path)
        script_dir = Path(__file__).resolve().parent
        repo_path = script_dir.parent / "repo" / "helm-gpt"

        if not repo_path.exists():
            print(f"Error: HELM-GPT repository not found at {repo_path}")
            return False

        sys.path.insert(0, str(repo_path))

        # Change to repo directory (required for data file access)
        original_cwd = os.getcwd()
        os.chdir(str(repo_path))

        try:
            import torch
            from model.model import load_gpt_model
            from agent.agent_trainer import AgentTrainer
            from utils.dataset import HelmDictionary

            _repo_modules = {
                'torch': torch,
                'load_gpt_model': load_gpt_model,
                'AgentTrainer': AgentTrainer,
                'HelmDictionary': HelmDictionary,
                'repo_path': repo_path
            }
            _repo_loaded = True
            return True

        finally:
            os.chdir(original_cwd)

    except ImportError as e:
        print(f"Error importing HELM-GPT modules: {e}")
        print("This script requires the HELM-GPT library and Python 3.7 environment.")
        return False
    except Exception as e:
        print(f"Error loading repository: {e}")
        return False


def _get_device(device_pref: str) -> str:
    """Determine the device to use for training."""
    if not _load_repo_modules():
        return "cpu"

    torch = _repo_modules['torch']

    if device_pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_pref == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        return "cpu"
    return device_pref


# ==============================================================================
# Core Functions
# ==============================================================================
def run_optimize_peptides(
    prior_model: Union[str, Path],
    task: str,
    output_dir: Union[str, Path],
    agent_model: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Optimize cyclic peptides for specific properties using RL-based agent training.

    Args:
        prior_model: Path to pre-trained prior model (.pt file)
        task: Optimization task ('permeability', 'kras_kd', or 'kras_perm')
        output_dir: Directory to save results and checkpoints
        agent_model: Optional path to existing agent model for continued training
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - status: "success" or "error"
            - output_dir: Path to output directory
            - checkpoints: List of saved checkpoint paths
            - final_model: Path to final agent model
            - sample_sequences: Generated optimized sequences
            - statistics: Training and sequence statistics
            - metadata: Execution metadata

    Example:
        >>> result = run_optimize_peptides("prior.pt", "permeability", "./output")
        >>> print(f"Best score: {result['statistics']['best_score']:.3f}")
    """
    # Validate inputs
    prior_model = Path(prior_model)
    output_dir = Path(output_dir)

    if not prior_model.exists():
        return {
            "status": "error",
            "error": f"Prior model not found: {prior_model}"
        }

    if task not in VALID_TASKS:
        return {
            "status": "error",
            "error": f"Invalid task '{task}'. Must be one of: {VALID_TASKS}"
        }

    if agent_model:
        agent_model = Path(agent_model)
        if not agent_model.exists():
            return {
                "status": "error",
                "error": f"Agent model not found: {agent_model}"
            }

    # Load repository modules
    if not _load_repo_modules():
        return {
            "status": "error",
            "error": "Failed to load HELM-GPT repository. Ensure Python 3.7 environment is active."
        }

    # Merge config
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Get device
    device = _get_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get modules
    torch = _repo_modules['torch']
    load_gpt_model = _repo_modules['load_gpt_model']
    AgentTrainer = _repo_modules['AgentTrainer']
    HelmDictionary = _repo_modules['HelmDictionary']
    repo_path = _repo_modules['repo_path']

    # Change to repo directory for model loading
    original_cwd = os.getcwd()
    os.chdir(str(repo_path))

    try:
        # Load prior model
        print(f"Loading prior model from {prior_model}...")
        model_def = prior_model.with_suffix('.json')
        prior = load_gpt_model(model_def, str(prior_model), device, copy_to_cpu=False)
        print("Prior model loaded successfully")

        # Load or copy agent model
        if agent_model:
            print(f"Loading existing agent model from {agent_model}...")
            agent = load_gpt_model(model_def, str(agent_model), device, copy_to_cpu=False)
        else:
            print("Creating new agent model from prior...")
            agent = copy.deepcopy(prior)

        # Setup scoring functions based on task
        score_type = 'weight'
        if task == 'permeability':
            score_fns = ['permeability']
            score_weights = [1.0]
        elif task == 'kras_kd':
            score_fns = ['kras_kd']
            score_weights = [1.0]
        elif task == 'kras_perm':
            score_fns = ['permeability', 'kras_kd']
            score_weights = [1.0, 1.0]
            score_type = 'sum'

        print(f"Task: {task}")
        print(f"Scoring functions: {score_fns}")
        print(f"Weights: {score_weights}")

        # Initialize trainer
        print("Initializing agent trainer...")
        trainer = AgentTrainer(
            prior_model=prior,
            agent_model=agent,
            save_dir=str(output_dir),
            device=device,
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            n_steps=config['n_steps'],
            sigma=config['sigma'],
            score_type=score_type,
            max_seq_len=config['max_len'],
            score_fns=score_fns,
            score_weights=score_weights,
            save_per_n_steps=config['save_every_n_steps'],
            loss_type=config['loss_type'],
            alpha=config['alpha'],
            use_amp=config['use_amp']
        )

        # Save configuration
        run_config = {
            "task": task,
            "prior_model": str(prior_model),
            "agent_model": str(agent_model) if agent_model else None,
            "output_dir": str(output_dir),
            **config,
            "device": device,
            "started_at": datetime.now().isoformat()
        }

        config_path = output_dir / "optimization_config.json"
        with open(config_path, 'w') as f:
            json.dump(run_config, f, indent=2)
        print(f"Configuration saved to {config_path}")

        # Train agent
        print(f"\nStarting optimization for {config['n_steps']} steps...")
        print(f"Batch size: {config['batch_size']}, Sigma: {config['sigma']}")
        print(f"Loss type: {config['loss_type']}, Alpha: {config['alpha']}")
        print("-" * 50)

        trainer.train()

        print("-" * 50)
        print("Optimization completed!")

        # Collect results
        checkpoints = list(output_dir.glob("Agent_*.pt"))
        final_model = None
        for cp in checkpoints:
            if "final" in cp.name:
                final_model = cp
                break

        if not final_model and checkpoints:
            final_model = max(checkpoints, key=lambda p: p.stat().st_mtime)

        # Collect step data
        step_files = list(output_dir.glob("step_*_aa_seqs.csv"))
        all_sequences = []
        all_scores = []

        for step_file in sorted(step_files):
            try:
                df = pd.read_csv(step_file)
                if 'helm_seqs' in df.columns:
                    all_sequences.extend(df['helm_seqs'].tolist())
                if 'weight' in df.columns:
                    all_scores.extend(df['weight'].tolist())
                elif 'sum' in df.columns:
                    all_scores.extend(df['sum'].tolist())
            except Exception:
                pass

        # Calculate statistics
        statistics = {
            "total_steps": config['n_steps'],
            "total_sequences_generated": len(all_sequences),
            "unique_sequences": len(set(all_sequences)) if all_sequences else 0
        }

        if all_scores:
            valid_scores = [s for s in all_scores if s is not None and not np.isnan(s)]
            if valid_scores:
                statistics.update({
                    "best_score": max(valid_scores),
                    "mean_score": np.mean(valid_scores),
                    "median_score": np.median(valid_scores),
                    "score_std": np.std(valid_scores)
                })

        # Get top sequences
        top_sequences = []
        if all_sequences and all_scores and len(all_sequences) == len(all_scores):
            seq_scores = list(zip(all_sequences, all_scores))
            seq_scores = [(s, sc) for s, sc in seq_scores if sc is not None and not np.isnan(sc)]
            seq_scores.sort(key=lambda x: x[1], reverse=True)

            seen = set()
            for seq, score in seq_scores[:50]:
                if seq not in seen:
                    top_sequences.append({
                        "helm_sequence": seq,
                        "score": score
                    })
                    seen.add(seq)
                    if len(top_sequences) >= 10:
                        break

        # Save top sequences
        if top_sequences:
            top_df = pd.DataFrame(top_sequences)
            top_path = output_dir / "top_sequences.csv"
            top_df.to_csv(top_path, index=False)
            print(f"\nTop sequences saved to {top_path}")

            print("\nTop 5 optimized sequences:")
            for i, seq_info in enumerate(top_sequences[:5]):
                print(f"  {i+1}. Score: {seq_info['score']:.4f}")
                print(f"     HELM: {seq_info['helm_sequence'][:80]}...")

        run_config["completed_at"] = datetime.now().isoformat()
        with open(config_path, 'w') as f:
            json.dump(run_config, f, indent=2)

        return {
            "status": "success",
            "output_dir": str(output_dir),
            "checkpoints": [str(cp) for cp in checkpoints],
            "final_model": str(final_model) if final_model else None,
            "config_file": str(config_path),
            "top_sequences": top_sequences,
            "statistics": statistics,
            "metadata": {
                "task": task,
                "prior_model": str(prior_model),
                "device": device,
                "config": config
            }
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    finally:
        os.chdir(original_cwd)


def check_optimization_requirements() -> Dict[str, Any]:
    """
    Check if all requirements for optimization are met.

    Returns:
        Dict with availability status for each requirement
    """
    script_dir = Path(__file__).parent
    mcp_root = script_dir.parent
    repo_path = mcp_root / "repo" / "helm-gpt"
    models_dir = mcp_root / "examples" / "data" / "models"

    requirements = {
        "repo_available": repo_path.exists(),
        "repo_path": str(repo_path),
        "permeability_model": (models_dir / "regression_rf.pkl").exists(),
        "kras_model": (models_dir / "kras_xgboost_reg.pkl").exists(),
        "models_directory": str(models_dir),
        "cuda_available": False,
        "python_version": sys.version
    }

    try:
        import torch
        requirements["cuda_available"] = torch.cuda.is_available()
        requirements["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            requirements["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        requirements["torch_version"] = "Not installed"

    # Check for prior models
    prior_models = list(repo_path.glob("result/prior/*.pt")) if repo_path.exists() else []
    requirements["available_prior_models"] = [str(p) for p in prior_models]

    return requirements


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--prior', '-p', required=True,
                        help='Path to pre-trained prior model (.pt file)')
    parser.add_argument('--task', '-t', required=True,
                        choices=VALID_TASKS,
                        help='Optimization task: permeability, kras_kd, or kras_perm')
    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for results and checkpoints')

    # Optional arguments
    parser.add_argument('--agent', '-a',
                        help='Path to existing agent model for continued training')
    parser.add_argument('--config', '-c',
                        help='Configuration file (JSON)')

    # Training parameters
    parser.add_argument('--n_steps', type=int, default=DEFAULT_CONFIG['n_steps'],
                        help=f'Number of training steps (default: {DEFAULT_CONFIG["n_steps"]})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help=f'Batch size (default: {DEFAULT_CONFIG["batch_size"]})')
    parser.add_argument('--sigma', type=float, default=DEFAULT_CONFIG['sigma'],
                        help=f'Augmented likelihood sigma (default: {DEFAULT_CONFIG["sigma"]})')
    parser.add_argument('--learning_rate', '--lr', type=float,
                        default=DEFAULT_CONFIG['learning_rate'],
                        help=f'Learning rate (default: {DEFAULT_CONFIG["learning_rate"]})')
    parser.add_argument('--max_len', type=int, default=DEFAULT_CONFIG['max_len'],
                        help=f'Maximum sequence length (default: {DEFAULT_CONFIG["max_len"]})')

    # Loss function parameters
    parser.add_argument('--loss_type', default=DEFAULT_CONFIG['loss_type'],
                        choices=['reinvent', 'cpl', 'reinvent_cpl'],
                        help=f'Loss function type (default: {DEFAULT_CONFIG["loss_type"]})')
    parser.add_argument('--alpha', type=float, default=DEFAULT_CONFIG['alpha'],
                        help=f'Alpha parameter for CPL loss (default: {DEFAULT_CONFIG["alpha"]})')

    # Device
    parser.add_argument('--device', default=DEFAULT_CONFIG['device'],
                        choices=['auto', 'cpu', 'cuda'],
                        help=f'Device for training (default: {DEFAULT_CONFIG["device"]})')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')

    # Utility
    parser.add_argument('--check', action='store_true',
                        help='Check requirements and exit')

    args = parser.parse_args()

    # Check requirements if requested
    if args.check:
        requirements = check_optimization_requirements()
        print("Optimization Requirements Check")
        print("=" * 40)
        for key, value in requirements.items():
            print(f"{key}: {value}")
        return 0

    # Load config file if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Build config overrides from CLI arguments
    config_overrides = {}
    if args.n_steps != DEFAULT_CONFIG['n_steps']:
        config_overrides['n_steps'] = args.n_steps
    if args.batch_size != DEFAULT_CONFIG['batch_size']:
        config_overrides['batch_size'] = args.batch_size
    if args.sigma != DEFAULT_CONFIG['sigma']:
        config_overrides['sigma'] = args.sigma
    if args.learning_rate != DEFAULT_CONFIG['learning_rate']:
        config_overrides['learning_rate'] = args.learning_rate
    if args.max_len != DEFAULT_CONFIG['max_len']:
        config_overrides['max_len'] = args.max_len
    if args.loss_type != DEFAULT_CONFIG['loss_type']:
        config_overrides['loss_type'] = args.loss_type
    if args.alpha != DEFAULT_CONFIG['alpha']:
        config_overrides['alpha'] = args.alpha
    if args.device != DEFAULT_CONFIG['device']:
        config_overrides['device'] = args.device
    if args.no_amp:
        config_overrides['use_amp'] = False

    # Run optimization
    try:
        result = run_optimize_peptides(
            prior_model=args.prior,
            task=args.task,
            output_dir=args.output_dir,
            agent_model=args.agent,
            config=config,
            **config_overrides
        )

        if result['status'] == 'success':
            print(f"\nOptimization completed successfully!")
            print(f"Output directory: {result['output_dir']}")
            if result.get('final_model'):
                print(f"Final model: {result['final_model']}")
            if result.get('statistics'):
                stats = result['statistics']
                print(f"\nStatistics:")
                print(f"  Total sequences: {stats.get('total_sequences_generated', 'N/A')}")
                print(f"  Unique sequences: {stats.get('unique_sequences', 'N/A')}")
                if 'best_score' in stats:
                    print(f"  Best score: {stats['best_score']:.4f}")
                    print(f"  Mean score: {stats['mean_score']:.4f}")
            return 0
        else:
            print(f"\nOptimization failed: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(f"\nTraceback:\n{result['traceback']}")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
