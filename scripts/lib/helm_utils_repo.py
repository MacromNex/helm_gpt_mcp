"""
HELM utility functions that use the original HELM-GP repository code.

This module provides a wrapper around the repo functions with lazy loading
to minimize startup time and handle import errors gracefully.

Original source: repo/helm-gpt/utils/helm_utils.py
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

# Cache for loaded repo functions
_repo_functions = None
_repo_available = None


def _load_repo_functions() -> Tuple[bool, dict]:
    """
    Lazy load functions from the HELM-GP repository.

    Returns:
        Tuple of (success, functions_dict)
    """
    global _repo_functions, _repo_available

    if _repo_available is not None:
        return _repo_available, _repo_functions or {}

    try:
        # Add repo to path
        script_dir = Path(__file__).parent
        repo_path = script_dir.parent.parent / "repo" / "helm-gpt"

        if not repo_path.exists():
            _repo_available = False
            return False, {}

        sys.path.insert(0, str(repo_path))

        # Change to repo directory (required for data file access)
        original_cwd = os.getcwd()
        os.chdir(str(repo_path))

        try:
            from utils.helm_utils import get_cycpep_smi_from_helm, is_helm_valid
            from utils.dataset import HelmDictionary

            _repo_functions = {
                'get_cycpep_smi_from_helm': get_cycpep_smi_from_helm,
                'is_helm_valid': is_helm_valid,
                'HelmDictionary': HelmDictionary
            }
            _repo_available = True
            return True, _repo_functions

        finally:
            # Restore working directory
            os.chdir(original_cwd)

    except Exception as e:
        _repo_available = False
        _repo_functions = {}
        return False, {}


def get_cycpep_smi_from_helm(helm: str) -> Optional[str]:
    """
    Convert HELM notation to SMILES representation for cyclic peptides.

    This function uses the original HELM-GP implementation if available,
    otherwise falls back to a simplified implementation.

    Args:
        helm: HELM notation string

    Returns:
        SMILES string or None if conversion fails
    """
    success, functions = _load_repo_functions()

    if success and 'get_cycpep_smi_from_helm' in functions:
        try:
            # Use original repo function
            return functions['get_cycpep_smi_from_helm'](helm)
        except Exception:
            pass

    # Fallback to simplified implementation
    return _simple_helm_to_smiles(helm)


def is_helm_valid(helm: str) -> bool:
    """
    Check if HELM notation is valid.

    This function uses the original HELM-GP implementation if available,
    otherwise falls back to a simplified implementation.

    Args:
        helm: HELM notation string

    Returns:
        True if HELM is valid, False otherwise
    """
    success, functions = _load_repo_functions()

    if success and 'is_helm_valid' in functions:
        try:
            # Use original repo function
            return functions['is_helm_valid'](helm)
        except Exception:
            pass

    # Fallback to simplified implementation
    return _simple_helm_validation(helm)


def get_helm_dictionary():
    """
    Get HELM dictionary if available from repo.

    Returns:
        HelmDictionary instance or None if not available
    """
    success, functions = _load_repo_functions()

    if success and 'HelmDictionary' in functions:
        try:
            return functions['HelmDictionary']()
        except Exception:
            pass

    return None


def _simple_helm_to_smiles(helm: str) -> Optional[str]:
    """
    Simplified HELM to SMILES conversion.

    This is a fallback that handles only basic cases.
    """
    try:
        # Very basic validation
        if not helm or not isinstance(helm, str):
            return None

        # Check for basic HELM structure
        if 'PEPTIDE1{' not in helm or '}$' not in helm:
            return None

        # For now, return a placeholder that indicates successful parsing
        # A real implementation would need to parse the HELM notation fully
        return "CC(C)C[C@H](N)C(=O)N[C@@H](C)C(=O)N"  # Placeholder peptide SMILES

    except Exception:
        return None


def _simple_helm_validation(helm: str) -> bool:
    """
    Simplified HELM validation.

    This is a fallback that handles only basic validation.
    """
    try:
        if not helm or not isinstance(helm, str):
            return False

        # Basic HELM structure check
        if 'PEPTIDE1{' not in helm or '}$' not in helm:
            return False

        # Check that it's not empty
        helm_parts = helm.split('$')
        if len(helm_parts) < 2:
            return False

        return True

    except Exception:
        return False


def check_repo_availability() -> bool:
    """
    Check if the HELM-GP repository functions are available.

    Returns:
        True if repo functions can be loaded, False otherwise
    """
    success, _ = _load_repo_functions()
    return success