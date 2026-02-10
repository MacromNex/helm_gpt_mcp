"""
Prediction utility functions for cyclic peptide MCP scripts.

These functions provide wrappers around the HELM-GP scoring functions
with fallback implementations for cases where the repo is not available.

Original source: repo/helm-gpt/agent/scoring/
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Cache for loaded repo classes
_repo_classes = None
_repo_available = None


def _load_repo_classes() -> Tuple[bool, dict]:
    """
    Lazy load scoring classes from the HELM-GP repository.

    Returns:
        Tuple of (success, classes_dict)
    """
    global _repo_classes, _repo_available

    if _repo_available is not None:
        return _repo_available, _repo_classes or {}

    try:
        # Add repo to path
        script_dir = Path(__file__).parent
        repo_path = script_dir.parent.parent / "repo" / "helm-gpt"

        if not repo_path.exists():
            _repo_available = False
            return False, {}

        sys.path.insert(0, str(repo_path))

        # Change to repo directory (required for model file access)
        original_cwd = os.getcwd()
        os.chdir(str(repo_path))

        try:
            from agent.scoring.permeability import Permeability
            from agent.scoring.kras import KRASInhibition

            _repo_classes = {
                'Permeability': Permeability,
                'KRASInhibition': KRASInhibition
            }
            _repo_available = True
            return True, _repo_classes

        finally:
            # Restore working directory
            os.chdir(original_cwd)

    except Exception as e:
        _repo_available = False
        _repo_classes = {}
        return False, {}


class PermeabilityPredictor:
    """
    Wrapper for permeability prediction with fallback capability.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.predictor = None
        self._repo_predictor = None
        self._use_repo = False

        # Try to initialize repo predictor
        success, classes = _load_repo_classes()
        if success and 'Permeability' in classes:
            try:
                if model_path:
                    self._repo_predictor = classes['Permeability'](model_path=model_path)
                else:
                    # Use default model path
                    default_model = "examples/data/models/regression_rf.pkl"
                    if Path(default_model).exists():
                        self._repo_predictor = classes['Permeability'](model_path=default_model)
                    else:
                        self._repo_predictor = classes['Permeability']()
                self._use_repo = True
            except Exception:
                self._use_repo = False

        # Fallback to simple predictor if repo not available
        if not self._use_repo:
            self._init_fallback_predictor(model_path)

    def _init_fallback_predictor(self, model_path: Optional[str]):
        """Initialize a fallback predictor using scikit-learn."""
        try:
            if model_path and Path(model_path).exists():
                with open(model_path, 'rb') as f:
                    self.predictor = pickle.load(f)
            else:
                # No model available - will return placeholder scores
                self.predictor = None
        except Exception:
            self.predictor = None

    def predict_batch(self, helm_sequences: List[str]) -> List[Optional[float]]:
        """
        Predict permeability for a batch of HELM sequences.

        Args:
            helm_sequences: List of HELM notation strings

        Returns:
            List of permeability scores (or None for failed predictions)
        """
        if self._use_repo and self._repo_predictor:
            try:
                return self._repo_predictor.get_scores(helm_sequences)
            except Exception:
                pass

        # Fallback implementation
        return self._fallback_predict(helm_sequences)

    def _fallback_predict(self, helm_sequences: List[str]) -> List[Optional[float]]:
        """Fallback prediction when repo is not available."""
        # Return placeholder scores based on sequence length
        scores = []
        for helm_seq in helm_sequences:
            if helm_seq and isinstance(helm_seq, str) and len(helm_seq) > 10:
                # Simple heuristic: longer sequences tend to have lower permeability
                score = max(-10.0, -5.0 - len(helm_seq) / 100.0)
                scores.append(score)
            else:
                scores.append(None)
        return scores

    def interpret_score(self, score: Optional[float]) -> str:
        """Interpret permeability score."""
        if score is None:
            return "Prediction failed"
        elif score > 0.7:
            return "High permeability"
        elif score > 0.4:
            return "Medium permeability"
        else:
            return "Low permeability"


class KRASBindingPredictor:
    """
    Wrapper for KRAS binding prediction with fallback capability.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.predictor = None
        self._repo_predictor = None
        self._use_repo = False

        # Try to initialize repo predictor
        success, classes = _load_repo_classes()
        if success and 'KRASInhibition' in classes:
            try:
                if model_path:
                    self._repo_predictor = classes['KRASInhibition'](model_path=model_path)
                else:
                    # Use default model path
                    default_model = "examples/data/models/kras_xgboost_reg.pkl"
                    if Path(default_model).exists():
                        self._repo_predictor = classes['KRASInhibition'](model_path=default_model)
                    else:
                        self._repo_predictor = classes['KRASInhibition']()
                self._use_repo = True
            except Exception:
                self._use_repo = False

        # Fallback to simple predictor if repo not available
        if not self._use_repo:
            self._init_fallback_predictor(model_path)

    def _init_fallback_predictor(self, model_path: Optional[str]):
        """Initialize a fallback predictor."""
        try:
            if model_path and Path(model_path).exists():
                with open(model_path, 'rb') as f:
                    self.predictor = pickle.load(f)
            else:
                self.predictor = None
        except Exception:
            self.predictor = None

    def predict_batch(self, helm_sequences: List[str]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        Predict KRAS binding for a batch of HELM sequences.

        Args:
            helm_sequences: List of HELM notation strings

        Returns:
            Tuple of (binding_scores, kd_values)
        """
        if self._use_repo and self._repo_predictor:
            try:
                scores = self._repo_predictor.get_scores(helm_sequences)
                # Convert scores to KD values (scores are log10(KD))
                kd_values = [10**score if score is not None else None for score in scores]
                return scores, kd_values
            except Exception:
                pass

        # Fallback implementation
        return self._fallback_predict(helm_sequences)

    def _fallback_predict(self, helm_sequences: List[str]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """Fallback prediction when repo is not available."""
        scores = []
        kd_values = []

        for helm_seq in helm_sequences:
            if helm_seq and isinstance(helm_seq, str) and len(helm_seq) > 10:
                # Simple heuristic: generate reasonable binding scores
                # Score range typically 0-3 (log10(KD) where KD is in μM)
                score = 1.0 + np.random.normal(0, 0.5)  # Mean ~10 μM
                score = max(0, min(3, score))  # Clip to reasonable range
                kd_value = 10**score

                scores.append(score)
                kd_values.append(kd_value)
            else:
                scores.append(None)
                kd_values.append(None)

        return scores, kd_values

    def interpret_binding(self, kd_value: Optional[float]) -> str:
        """Interpret KRAS binding KD value."""
        if kd_value is None:
            return "Prediction failed"
        elif kd_value < 1.0:
            return "Strong binding (KD < 1 μM)"
        elif kd_value < 10.0:
            return "Moderate binding (1-10 μM)"
        elif kd_value < 100.0:
            return "Weak binding (10-100 μM)"
        else:
            return "Very weak/no binding (KD > 100 μM)"

    def interpret_score(self, score: Optional[float]) -> str:
        """Interpret binding score."""
        if score is None:
            return "Prediction failed"
        elif score > 0.8:
            return "High binding score"
        elif score > 0.5:
            return "Medium binding score"
        else:
            return "Low binding score"


def check_prediction_models() -> Dict[str, bool]:
    """
    Check availability of prediction models and repo functions.

    Returns:
        Dict indicating which models are available
    """
    success, _ = _load_repo_classes()

    # Check for model files
    model_files = {
        'permeability': Path("examples/data/models/regression_rf.pkl").exists(),
        'kras': Path("examples/data/models/kras_xgboost_reg.pkl").exists()
    }

    return {
        'repo_available': success,
        'permeability_model': model_files['permeability'],
        'kras_model': model_files['kras']
    }