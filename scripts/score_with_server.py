#!/usr/bin/env python3
"""
Script: score_with_server.py
Description: Score cyclic peptides using external Boltz2 or Rosetta servers.

This script provides a unified interface to score peptide sequences using
server-based computational methods like structure prediction (Boltz2) and
energy calculations (Rosetta).

Dependencies: requests, pandas, yaml, easydict (optional)

Usage:
    # Score single sequence
    python scripts/score_with_server.py \
        --input "ACDEFGHIK" \
        --scorer rosetta \
        --server http://localhost:8001

    # Score batch from CSV
    python scripts/score_with_server.py \
        --input sequences.csv \
        --output scores.csv \
        --scorer boltz2 \
        --config configs/boltz2_config.yaml

    # Use custom target scores
    python scripts/score_with_server.py \
        --input sequences.csv \
        --scorer rosetta \
        --target-scores '{"ddG": {"weight": 1.0}, "SAP": {"weight": 0.5}}'
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import pandas as pd
import yaml

# Add local lib to path
sys.path.insert(0, str(Path(__file__).parent))
from lib.server_scoring_utils import (
    ServerScorerWrapper,
    get_available_server_scorers,
    check_server_connection
)

# Default configuration
DEFAULT_CONFIG = {
    'input': {
        'helm_column': 'HELM',
        'sequence_column': 'sequence',
        'encoding': 'utf-8'
    },
    'processing': {
        'batch_size': 10,
        'skip_invalid': True,
        'limit': None
    },
    'output': {
        'include_raw_scores': True,
        'include_structures': False,
        'include_metadata': True
    }
}


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_server_scoring(
    input_file: Union[str, Path],
    scorer_type: str,
    output_file: Optional[Union[str, Path]] = None,
    server_host: Optional[str] = None,
    server_api: Optional[str] = None,
    server_config: Optional[Dict[str, Any]] = None,
    target_scores: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    helm_column: str = 'HELM',
    sequence_column: str = 'sequence',
    limit: Optional[int] = None,
    include_raw_scores: bool = True,
    include_structures: bool = False,
    timeout: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Score peptides using external server (Boltz2/Rosetta).

    Args:
        input_file: Path to CSV with sequences, or single sequence string
        scorer_type: 'boltz2' or 'rosetta'
        output_file: Optional path to save results CSV
        server_host: Server URL (e.g., 'http://localhost:8001')
        server_api: API endpoint (e.g., '/rosetta/score')
        server_config: Full server configuration dict
        target_scores: Dict of {score_name: {weight, transform}}
        config: Override processing configuration
        helm_column: Column name for HELM sequences in CSV
        sequence_column: Column name for amino acid sequences in CSV
        limit: Maximum sequences to process
        include_raw_scores: Include raw scores before transformation
        include_structures: Include predicted structures in output
        timeout: Request timeout in seconds
        **kwargs: Additional configuration overrides

    Returns:
        Dictionary with scoring results:
        - status: 'success', 'partial', or 'error'
        - results: List of {sequence, score, raw_scores, ...}
        - summary: Statistics about scoring
        - output_file: Path to saved CSV (if requested)
    """
    # Merge configurations
    cfg = {**DEFAULT_CONFIG}
    if config:
        cfg = _deep_merge(cfg, config)
    if kwargs:
        cfg = _deep_merge(cfg, kwargs)

    # Override with explicit parameters
    if helm_column:
        cfg['input']['helm_column'] = helm_column
    if sequence_column:
        cfg['input']['sequence_column'] = sequence_column
    if limit is not None:
        cfg['processing']['limit'] = limit
    cfg['output']['include_raw_scores'] = include_raw_scores
    cfg['output']['include_structures'] = include_structures

    # Build scorer configuration
    scorer_config = server_config or {}
    if server_host:
        scorer_config.setdefault('server', {})['host'] = server_host
    if server_api:
        scorer_config.setdefault('server', {})['api'] = server_api
    if timeout:
        scorer_config.setdefault('server', {})['timeout'] = timeout
    if target_scores:
        scorer_config['target_scores'] = target_scores

    # Load sequences
    input_path = Path(input_file) if isinstance(input_file, str) else input_file

    if isinstance(input_file, str) and not input_path.exists():
        # Treat as single sequence
        sequences = [input_file]
        is_single = True
    elif input_path.suffix.lower() == '.csv':
        df = pd.read_csv(input_path, encoding=cfg['input']['encoding'])

        # Find sequence column
        seq_col = None
        if cfg['input']['helm_column'] in df.columns:
            seq_col = cfg['input']['helm_column']
        elif cfg['input']['sequence_column'] in df.columns:
            seq_col = cfg['input']['sequence_column']
        else:
            # Try common column names
            for col in ['HELM', 'helm', 'sequence', 'Sequence', 'seq', 'peptide']:
                if col in df.columns:
                    seq_col = col
                    break

        if seq_col is None:
            raise ValueError(f"No sequence column found. Tried: {cfg['input']['helm_column']}, "
                           f"{cfg['input']['sequence_column']}")

        sequences = df[seq_col].dropna().tolist()
        is_single = False
    else:
        # Try to read as text file with one sequence per line
        with open(input_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        is_single = len(sequences) == 1

    # Apply limit
    if cfg['processing']['limit']:
        sequences = sequences[:cfg['processing']['limit']]

    # Initialize scorer
    try:
        scorer = ServerScorerWrapper(
            scorer_type=scorer_type,
            config=scorer_config,
            server_host=server_host,
            server_api=server_api,
            timeout=timeout
        )
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Failed to initialize {scorer_type} scorer: {str(e)}',
            'scorer_type': scorer_type
        }

    # Score sequences
    scoring_result = scorer.score(
        sequences=sequences,
        return_raw_scores=include_raw_scores
    )

    # Build results
    results = []
    for i, seq in enumerate(sequences):
        result_entry = {
            'sequence': seq,
            'score': scoring_result['scores'][i] if i < len(scoring_result['scores']) else None
        }

        if include_raw_scores and scoring_result.get('raw_scores'):
            for score_name, values in scoring_result['raw_scores'].items():
                if i < len(values):
                    result_entry[f'raw_{score_name}'] = values[i]

        if include_structures and scoring_result.get('structures'):
            if i < len(scoring_result['structures']):
                result_entry['structure'] = scoring_result['structures'][i]

        results.append(result_entry)

    # Calculate summary statistics
    valid_scores = [r['score'] for r in results if r['score'] is not None]

    summary = {
        'total_sequences': len(sequences),
        'scored_sequences': len(valid_scores),
        'failed_sequences': len(sequences) - len(valid_scores),
        'scorer_type': scorer_type,
        'using_repo_implementation': scorer._use_repo
    }

    if valid_scores:
        import numpy as np
        summary.update({
            'mean_score': float(np.mean(valid_scores)),
            'std_score': float(np.std(valid_scores)),
            'min_score': float(np.min(valid_scores)),
            'max_score': float(np.max(valid_scores))
        })

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create DataFrame for output
        output_df = pd.DataFrame(results)

        # Remove structure column if present (too large for CSV)
        if 'structure' in output_df.columns and not include_structures:
            output_df = output_df.drop(columns=['structure'])

        output_df.to_csv(output_path, index=False)

    # Build response
    response = {
        'status': scoring_result['status'],
        'scorer_type': scorer_type,
        'results': results if not is_single else results[0] if results else None,
        'summary': summary,
        'scorer_info': scorer.get_scorer_info()
    }

    if output_path:
        response['output_file'] = str(output_path)

    if scoring_result.get('errors'):
        response['errors'] = scoring_result['errors']

    return response


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Score cyclic peptides using external Boltz2 or Rosetta servers.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score single sequence with Rosetta
  python score_with_server.py --input "ACDEFGHIK" --scorer rosetta --server http://localhost:8001

  # Score batch from CSV with Boltz2
  python score_with_server.py --input sequences.csv --output scores.csv --scorer boltz2

  # Use YAML config for server settings
  python score_with_server.py --input sequences.csv --scorer rosetta --config rosetta_config.yaml
        """
    )

    parser.add_argument('--input', '-i',
                        help='Input CSV file with sequences, or single sequence string')
    parser.add_argument('--output', '-o',
                        help='Output CSV file path')
    parser.add_argument('--scorer', '-s', choices=['boltz2', 'rosetta'],
                        help='Scorer type (boltz2 or rosetta)')
    parser.add_argument('--server',
                        help='Server URL (e.g., http://localhost:8001)')
    parser.add_argument('--api',
                        help='API endpoint (e.g., /rosetta/score)')
    parser.add_argument('--config', '-c',
                        help='YAML config file with server and scoring settings')
    parser.add_argument('--target-scores',
                        help='JSON string of target scores configuration')
    parser.add_argument('--helm-column', default='HELM',
                        help='Column name for HELM sequences (default: HELM)')
    parser.add_argument('--sequence-column', default='sequence',
                        help='Column name for amino acid sequences (default: sequence)')
    parser.add_argument('--limit', type=int,
                        help='Maximum number of sequences to process')
    parser.add_argument('--timeout', type=int,
                        help='Request timeout in seconds')
    parser.add_argument('--include-structures', action='store_true',
                        help='Include predicted structures in output')
    parser.add_argument('--no-raw-scores', action='store_true',
                        help='Exclude raw scores from output')
    parser.add_argument('--check-connection', action='store_true',
                        help='Only check if server is reachable, then exit')
    parser.add_argument('--info', action='store_true',
                        help='Show available scorers and exit')

    args = parser.parse_args()

    # Show info and exit
    if args.info:
        info = get_available_server_scorers()
        print(json.dumps(info, indent=2))
        return 0

    # Validate required args for other operations
    if not args.input:
        if args.check_connection:
            if not args.server:
                print("Error: --server is required for connection check")
                return 1
        else:
            parser.error("--input is required for scoring")

    if not args.scorer and not args.check_connection and not args.info:
        parser.error("--scorer is required for scoring")

    # Check connection and exit
    if args.check_connection:
        if not args.server:
            print("Error: --server is required for connection check")
            return 1

        result = check_server_connection(
            scorer_type=args.scorer,
            server_host=args.server,
            server_api=args.api,
            timeout=args.timeout or 10
        )
        print(json.dumps(result, indent=2))
        return 0 if result['status'] in ['connected', 'reachable'] else 1

    # Load YAML config if provided
    yaml_config = None
    if args.config:
        try:
            yaml_config = load_yaml_config(args.config)
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1

    # Parse target scores if provided
    target_scores = None
    if args.target_scores:
        try:
            target_scores = json.loads(args.target_scores)
        except json.JSONDecodeError as e:
            print(f"Error parsing target-scores JSON: {e}")
            return 1

    # Run scoring
    try:
        result = run_server_scoring(
            input_file=args.input,
            scorer_type=args.scorer,
            output_file=args.output,
            server_host=args.server,
            server_api=args.api,
            server_config=yaml_config,
            target_scores=target_scores,
            helm_column=args.helm_column,
            sequence_column=args.sequence_column,
            limit=args.limit,
            timeout=args.timeout,
            include_raw_scores=not args.no_raw_scores,
            include_structures=args.include_structures
        )

        # Print result
        print(json.dumps(result, indent=2, default=str))

        # Return appropriate exit code
        if result['status'] == 'success':
            return 0
        elif result['status'] == 'partial':
            return 0  # Partial success is still success
        else:
            return 1

    except Exception as e:
        print(json.dumps({
            'status': 'error',
            'error': str(e)
        }, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
