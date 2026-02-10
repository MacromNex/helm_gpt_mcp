"""
Server scoring utility functions for cyclic peptide MCP scripts.

These functions provide wrappers around the HELM-GP server-based scoring functions
(Boltz2, Rosetta) with lazy loading and fallback capabilities.

Original source: repo/helm-gpt/agent/scoring/server/
"""

import sys
import os
import json
import requests
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Cache for loaded server scorer classes
_server_scorer_classes = None
_server_scorers_available = None


def _load_server_scorer_classes() -> Tuple[bool, dict]:
    """
    Lazy load server scorer classes from the HELM-GP repository.

    Returns:
        Tuple of (success, classes_dict)
    """
    global _server_scorer_classes, _server_scorers_available

    if _server_scorers_available is not None:
        return _server_scorers_available, _server_scorer_classes or {}

    try:
        # Add repo to path
        script_dir = Path(__file__).parent
        repo_path = script_dir.parent.parent / "repo" / "helm-gpt"

        if not repo_path.exists():
            _server_scorers_available = False
            return False, {}

        sys.path.insert(0, str(repo_path))

        # Change to repo directory (required for relative imports in repo)
        original_cwd = os.getcwd()
        os.chdir(str(repo_path))

        try:
            from agent.scoring.server import server_scorers
            from agent.scoring.server.base import BaseServerScorer
            from agent.scoring.server.boltz2 import Boltz2Scorer
            from agent.scoring.server.rosetta import RosettaScorer

            _server_scorer_classes = {
                'boltz2': Boltz2Scorer,
                'rosetta': RosettaScorer,
                'server_scorers': server_scorers,
                'BaseServerScorer': BaseServerScorer
            }
            _server_scorers_available = True
            return True, _server_scorer_classes

        finally:
            # Restore working directory
            os.chdir(original_cwd)

    except Exception as e:
        logger.warning(f"Server scorers not available: {e}")
        _server_scorers_available = False
        _server_scorer_classes = {}
        return False, {}


def _config_to_easydict(config: Dict[str, Any]) -> Any:
    """
    Convert a dictionary to EasyDict for use with server scorers.

    Args:
        config: Configuration dictionary

    Returns:
        EasyDict object (or plain dict if EasyDict not available)
    """
    try:
        from easydict import EasyDict
        return EasyDict(config)
    except ImportError:
        # Fallback to a simple namespace-like object that mimics EasyDict
        class ConfigNamespace:
            def __init__(self, d):
                self._data = d
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, ConfigNamespace(v))
                    else:
                        setattr(self, k, v)

            def get(self, key, default=None):
                return getattr(self, key, default)

            def keys(self):
                return self._data.keys()

            def values(self):
                return self._data.values()

            def items(self):
                return self._data.items()

            def __iter__(self):
                return iter(self._data)

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value
                if isinstance(value, dict):
                    setattr(self, key, ConfigNamespace(value))
                else:
                    setattr(self, key, value)

        return ConfigNamespace(config)


class ServerScorerWrapper:
    """
    Wrapper for server-based scoring with fallback capability.

    Supports Boltz2 and Rosetta scorers.
    """

    SUPPORTED_SCORERS = ['boltz2', 'rosetta']

    DEFAULT_CONFIGS = {
        'boltz2': {
            'server': {
                'host': 'http://localhost:8000',
                'api': '/biology/mit/boltz2/predict',
                'timeout': 300,
                'max_retries': 3
            },
            'task': {
                'seq_len': None,  # No length restriction
                'input_type': 'helm'
            },
            'boltz2': {
                'recycling_steps': 3,
                'sampling_steps': 200,
                'diffusion_samples': 1
            },
            'target_scores': {
                'iptm_scores': {
                    'weight': 1.0,
                    'transform': {
                        'trans_type': 'sigmoid',
                        'low': 0.5,
                        'high': 0.9,
                        'params': {'k': 2}
                    }
                }
            }
        },
        'rosetta': {
            'server': {
                'host': 'http://localhost:8001',
                'api': '/rosetta/score',
                'timeout': 600,
                'max_retries': 3
            },
            'task': {
                'seq_len': None,
                'input_type': 'helm',
                'chain_id': 'A',
                'interface_chains': ['A', 'B']
            },
            'rosetta': {
                'relax_structure': True,
                'minimize_sidechains': True,
                'pack_radius': 8.0,
                'repack_rounds': 3,
                'score_function': 'ref2015'
            },
            'target_scores': {
                'ddG': {
                    'weight': 1.0,
                    'transform': {
                        'trans_type': 'rsigmoid',
                        'low': -20.0,
                        'high': 0.0,
                        'params': {'k': 0.5}
                    }
                }
            }
        }
    }

    def __init__(
        self,
        scorer_type: str,
        config: Optional[Dict[str, Any]] = None,
        server_host: Optional[str] = None,
        server_api: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize server scorer wrapper.

        Args:
            scorer_type: 'boltz2' or 'rosetta'
            config: Full configuration dict (optional, uses defaults if not provided)
            server_host: Override server host URL
            server_api: Override API endpoint
            timeout: Override request timeout
        """
        if scorer_type not in self.SUPPORTED_SCORERS:
            raise ValueError(f"Unsupported scorer type: {scorer_type}. "
                           f"Must be one of: {self.SUPPORTED_SCORERS}")

        self.scorer_type = scorer_type
        self._scorer = None
        self._use_repo = False

        # Build configuration
        self.config = self._build_config(config, server_host, server_api, timeout)

        # Try to initialize repo scorer
        success, classes = _load_server_scorer_classes()
        if success and scorer_type in classes:
            try:
                easydict_config = _config_to_easydict(self.config)
                self._scorer = classes[scorer_type](easydict_config)
                self._use_repo = True
                logger.info(f"Initialized {scorer_type} scorer from repo")
            except Exception as e:
                logger.warning(f"Failed to initialize repo {scorer_type} scorer: {e}")
                self._use_repo = False

        if not self._use_repo:
            logger.info(f"Using fallback HTTP client for {scorer_type} scorer")

    def _build_config(
        self,
        config: Optional[Dict[str, Any]],
        server_host: Optional[str],
        server_api: Optional[str],
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """Build configuration by merging defaults with overrides."""
        # Start with default config
        result = dict(self.DEFAULT_CONFIGS[self.scorer_type])

        # Deep merge with provided config
        if config:
            result = self._deep_merge(result, config)

        # Apply overrides
        if server_host:
            result['server']['host'] = server_host
        if server_api:
            result['server']['api'] = server_api
        if timeout:
            result['server']['timeout'] = timeout

        return result

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def score(
        self,
        sequences: Union[str, List[str]],
        return_raw_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Score sequences using the server scorer.

        Args:
            sequences: Single sequence (HELM or AA) or list of sequences
            return_raw_scores: Include raw scores before transformation

        Returns:
            Dictionary with:
            - scores: Aggregated/transformed scores (list)
            - raw_scores: Raw scores per target (dict of lists)
            - structures: PDB/CIF structures if returned by server
            - errors: Any errors encountered
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        if self._use_repo and self._scorer:
            return self._score_with_repo(sequences, return_raw_scores)
        else:
            return self._score_with_fallback(sequences, return_raw_scores)

    def _score_with_repo(
        self,
        sequences: List[str],
        return_raw_scores: bool
    ) -> Dict[str, Any]:
        """Score using the repo scorer class."""
        try:
            # Use the scorer's __call__ method
            transformed_scores, raw_scores = self._scorer(sequences)

            result = {
                'status': 'success',
                'scorer_type': self.scorer_type,
                'scores': transformed_scores.tolist() if hasattr(transformed_scores, 'tolist') else list(transformed_scores),
                'sequence_count': len(sequences),
                'errors': []
            }

            if return_raw_scores:
                result['raw_scores'] = {
                    k: v.tolist() if hasattr(v, 'tolist') else list(v)
                    for k, v in raw_scores.items()
                }

            return result

        except Exception as e:
            logger.error(f"Repo scorer failed: {e}")
            return {
                'status': 'error',
                'scorer_type': self.scorer_type,
                'error': str(e),
                'scores': [None] * len(sequences),
                'sequence_count': len(sequences)
            }

    def _score_with_fallback(
        self,
        sequences: List[str],
        return_raw_scores: bool
    ) -> Dict[str, Any]:
        """Score using direct HTTP requests (fallback when repo not available)."""
        server_url = self.config['server']['host'].rstrip('/')
        api_endpoint = self.config['server']['api']
        timeout = self.config['server']['timeout']
        max_retries = self.config['server'].get('max_retries', 3)

        results = {
            'status': 'success',
            'scorer_type': self.scorer_type,
            'scores': [],
            'raw_scores': {} if return_raw_scores else None,
            'structures': [],
            'errors': [],
            'sequence_count': len(sequences)
        }

        for seq in sequences:
            try:
                # Build request data
                request_data = self._build_request_data(seq)

                # Make request with retries
                response = None
                last_error = None

                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            f"{server_url}{api_endpoint}",
                            json=request_data,
                            timeout=timeout
                        )
                        response.raise_for_status()
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            logger.warning(f"Retry {attempt + 1}/{max_retries} for sequence")

                if response is None:
                    raise last_error or Exception("No response received")

                # Parse response
                response_json = response.json()
                scores_dict, structure = self._extract_scores(response_json)

                # Aggregate scores (simple weighted sum for fallback)
                target_scores = self.config.get('target_scores', {})
                aggregated = 0.0
                for score_name, score_value in scores_dict.items():
                    if score_name in target_scores:
                        weight = target_scores[score_name].get('weight', 1.0)
                        aggregated += weight * (score_value or 0.0)

                results['scores'].append(aggregated)

                if return_raw_scores:
                    for k, v in scores_dict.items():
                        if k not in results['raw_scores']:
                            results['raw_scores'][k] = []
                        results['raw_scores'][k].append(v)

                if structure:
                    results['structures'].append(structure)

            except Exception as e:
                logger.error(f"Failed to score sequence: {e}")
                results['scores'].append(None)
                results['errors'].append({
                    'sequence': seq[:50] + '...' if len(seq) > 50 else seq,
                    'error': str(e)
                })

        # Update status if any errors
        if results['errors']:
            results['status'] = 'partial' if any(s is not None for s in results['scores']) else 'error'

        return results

    def _build_request_data(self, seq: str) -> Dict[str, Any]:
        """Build request payload for the server."""
        if self.scorer_type == 'boltz2':
            return self._build_boltz2_request(seq)
        elif self.scorer_type == 'rosetta':
            return self._build_rosetta_request(seq)
        else:
            return {'sequence': seq}

    def _build_boltz2_request(self, seq: str) -> Dict[str, Any]:
        """Build Boltz2 request payload."""
        boltz2_config = self.config.get('boltz2', {})
        task_config = self.config.get('task', {})

        data = {
            'polymers': [{'sequence': seq, 'type': 'protein'}],
            'ligands': [],
            'templates': [],
            'recycling_steps': boltz2_config.get('recycling_steps', 3),
            'sampling_steps': boltz2_config.get('sampling_steps', 200),
            'diffusion_samples': boltz2_config.get('diffusion_samples', 1)
        }

        # Add substrate if configured
        substrate = task_config.get('substrate')
        if substrate:
            data['affinity'] = {'binder': 'S'}
            data['polymers'].append({'sequence': substrate, 'type': 'protein', 'id': 'S'})

        return data

    def _build_rosetta_request(self, seq: str) -> Dict[str, Any]:
        """Build Rosetta request payload."""
        rosetta_config = self.config.get('rosetta', {})
        task_config = self.config.get('task', {})

        data = {
            'sequence': seq,
            'chain_id': task_config.get('chain_id', 'A'),
            'interface_chains': task_config.get('interface_chains', ['A', 'B']),
            'options': {
                'relax_structure': rosetta_config.get('relax_structure', True),
                'minimize_sidechains': rosetta_config.get('minimize_sidechains', True),
                'pack_radius': rosetta_config.get('pack_radius', 8.0),
                'repack_rounds': rosetta_config.get('repack_rounds', 3),
                'score_function': rosetta_config.get('score_function', 'ref2015')
            },
            'requested_scores': list(self.config.get('target_scores', {}).keys())
        }

        # Add target PDB if configured
        target_pdb = task_config.get('target_pdb')
        if target_pdb and Path(target_pdb).exists():
            with open(target_pdb, 'r') as f:
                data['target_pdb'] = f.read()

        return data

    def _extract_scores(self, response_json: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[str]]:
        """Extract scores from server response."""
        if self.scorer_type == 'boltz2':
            return self._extract_boltz2_scores(response_json)
        elif self.scorer_type == 'rosetta':
            return self._extract_rosetta_scores(response_json)
        else:
            return response_json.get('scores', {}), None

    def _extract_boltz2_scores(self, response: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[str]]:
        """Extract scores from Boltz2 response."""
        scores = {}
        target_scores = self.config.get('target_scores', {})

        for score_name in target_scores.keys():
            # Check direct score
            if score_name in response:
                val = response[score_name]
                scores[score_name] = val[0] if isinstance(val, list) else val
            # Check in affinities
            elif 'affinities' in response:
                affinities = response['affinities']
                if 'S' in affinities and score_name in affinities['S']:
                    val = affinities['S'][score_name]
                    scores[score_name] = val[0] if isinstance(val, list) else val

        # Extract structure
        structure = None
        if 'structures' in response and response['structures']:
            structure = response['structures'][0].get('structure')

        return scores, structure

    def _extract_rosetta_scores(self, response: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[str]]:
        """Extract scores from Rosetta response."""
        scores_dict = response.get('scores', response)
        target_scores = self.config.get('target_scores', {})

        scores = {}
        for score_name in target_scores.keys():
            if score_name in scores_dict:
                scores[score_name] = scores_dict[score_name]

        structure = response.get('pdb')

        return scores, structure

    def get_scorer_info(self) -> Dict[str, Any]:
        """Get information about this scorer."""
        return {
            'scorer_type': self.scorer_type,
            'using_repo_implementation': self._use_repo,
            'server_host': self.config['server']['host'],
            'server_api': self.config['server']['api'],
            'timeout': self.config['server']['timeout'],
            'target_scores': list(self.config.get('target_scores', {}).keys()),
            'config': self.config
        }


def get_available_server_scorers() -> Dict[str, Any]:
    """
    Get information about available server scorers.

    Returns:
        Dictionary with scorer availability and info
    """
    success, classes = _load_server_scorer_classes()

    return {
        'repo_available': success,
        'available_scorers': list(ServerScorerWrapper.SUPPORTED_SCORERS),
        'repo_scorers_loaded': list(classes.keys()) if success else [],
        'default_configs': ServerScorerWrapper.DEFAULT_CONFIGS,
        'supported_scores': {
            'boltz2': ['iptm_scores', 'ipae_scores', 'affinity_pic50', 'complex_ipde_scores'],
            'rosetta': ['ddG', 'SAP', 'CMS', 'total_score', 'interface_score']
        },
        'transformation_types': ['sigmoid', 'rsigmoid', 'dsigmoid']
    }


def check_server_connection(
    scorer_type: str,
    server_host: str,
    server_api: Optional[str] = None,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Check if a scoring server is reachable.

    Args:
        scorer_type: 'boltz2' or 'rosetta'
        server_host: Server URL
        server_api: Optional API endpoint to check
        timeout: Connection timeout in seconds

    Returns:
        Dictionary with connection status
    """
    if server_api is None:
        default_apis = {
            'boltz2': '/biology/mit/boltz2/predict',
            'rosetta': '/rosetta/score'
        }
        server_api = default_apis.get(scorer_type, '/health')

    try:
        # Try a simple GET request first (health check)
        health_url = f"{server_host.rstrip('/')}/health"
        try:
            response = requests.get(health_url, timeout=timeout)
            if response.ok:
                return {
                    'status': 'connected',
                    'server_host': server_host,
                    'health_check': 'passed'
                }
        except:
            pass

        # Try OPTIONS or HEAD on the API endpoint
        api_url = f"{server_host.rstrip('/')}{server_api}"
        response = requests.options(api_url, timeout=timeout)

        return {
            'status': 'reachable',
            'server_host': server_host,
            'api_endpoint': server_api,
            'response_code': response.status_code
        }

    except requests.exceptions.ConnectionError:
        return {
            'status': 'unreachable',
            'server_host': server_host,
            'error': 'Connection refused - server may not be running'
        }
    except requests.exceptions.Timeout:
        return {
            'status': 'timeout',
            'server_host': server_host,
            'error': f'Connection timed out after {timeout} seconds'
        }
    except Exception as e:
        return {
            'status': 'error',
            'server_host': server_host,
            'error': str(e)
        }
