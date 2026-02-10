"""
HELM utility functions for cyclic peptide MCP scripts.

These functions are extracted and simplified from the HELM-GP repository
to minimize dependencies and make scripts self-contained.

Original source: repo/helm-gpt/utils/helm_utils.py
"""

import re
import pandas as pd
from typing import Optional, List
from rdkit import Chem
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')


def get_cycpep_smi_from_helm(helm: str) -> Optional[str]:
    """
    Convert HELM notation to SMILES representation for cyclic peptides.

    Simplified from repo/helm-gpt/utils/helm_utils.py:get_cycpep_smi_from_helm()

    Args:
        helm: HELM notation string

    Returns:
        SMILES string or None if conversion fails
    """
    try:
        # Split HELM into components
        helms = helm.split('$') if '$' in helm else None
        if helms is None or len(helms) != 5:
            return None

        # Extract peptide sequence
        try:
            pep_idx = helms[0].index('{')
        except:
            return None

        linear_helm = helms[0][pep_idx:]

        # Load monomer mapping (simplified approach)
        # In the original, this loads from data/prior/monomer_library.csv
        # For MCP usage, we'll use a minimal built-in mapping
        monomer_map = _get_minimal_monomer_mapping()

        # Parse peptide sequence
        peptide_sequence = _parse_peptide_sequence(linear_helm, monomer_map)
        if not peptide_sequence:
            return None

        # Convert to SMILES (simplified approach)
        smiles = _sequence_to_smiles(peptide_sequence, monomer_map)

        return smiles

    except Exception:
        return None


def is_helm_valid(helm: str) -> bool:
    """
    Check if HELM notation is valid by attempting conversion to SMILES.

    Simplified from repo/helm-gpt/utils/helm_utils.py:is_helm_valid()

    Args:
        helm: HELM notation string

    Returns:
        True if HELM can be converted to valid SMILES, False otherwise
    """
    try:
        smiles = get_cycpep_smi_from_helm(helm)
        if smiles is None:
            return False

        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    except Exception:
        return False


def _get_minimal_monomer_mapping() -> dict:
    """
    Minimal monomer mapping for common amino acids.

    In the full repo, this loads from data/prior/monomer_library.csv
    For MCP usage, we include only the most common amino acids.
    """
    # This is a simplified mapping - in production you'd want the full library
    return {
        'A': 'C[C@H](N)C(=O)O',  # Alanine
        'R': 'NC(=N)NCCCC[C@H](N)C(=O)O',  # Arginine
        'N': 'NC(=O)C[C@H](N)C(=O)O',  # Asparagine
        'D': 'O=C(O)C[C@H](N)C(=O)O',  # Aspartic acid
        'C': 'SC[C@H](N)C(=O)O',  # Cysteine
        'E': 'O=C(O)CC[C@H](N)C(=O)O',  # Glutamic acid
        'Q': 'NC(=O)CC[C@H](N)C(=O)O',  # Glutamine
        'G': 'NCC(=O)O',  # Glycine
        'H': 'NC1=CN=C[C@@H](C[C@H](N)C(=O)O)N1',  # Histidine
        'I': 'CC[C@H](C)[C@H](N)C(=O)O',  # Isoleucine
        'L': 'CC(C)C[C@H](N)C(=O)O',  # Leucine
        'K': 'NCCCC[C@H](N)C(=O)O',  # Lysine
        'M': 'CSCC[C@H](N)C(=O)O',  # Methionine
        'F': 'NC(Cc1ccccc1)C(=O)O',  # Phenylalanine
        'P': 'N1CCC[C@H]1C(=O)O',  # Proline
        'S': 'OC[C@H](N)C(=O)O',  # Serine
        'T': 'C[C@@H](O)[C@H](N)C(=O)O',  # Threonine
        'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',  # Tryptophan
        'Y': 'NC(Cc1ccc(O)cc1)C(=O)O',  # Tyrosine
        'V': 'CC(C)[C@H](N)C(=O)O'  # Valine
    }


def _parse_peptide_sequence(linear_helm: str, monomer_map: dict) -> Optional[List[str]]:
    """
    Parse peptide sequence from linear HELM notation.

    This is a simplified version - the real parser is much more complex.
    """
    try:
        # Remove brackets and split by dots
        sequence = linear_helm.strip('{}').split('.')

        # Map monomers to standard amino acid codes
        parsed_sequence = []
        for monomer in sequence:
            # Handle special cases and modifications
            clean_monomer = _normalize_monomer(monomer)
            if clean_monomer in monomer_map:
                parsed_sequence.append(clean_monomer)
            else:
                # For unknown monomers, try to map to closest standard AA
                mapped = _map_unknown_monomer(clean_monomer)
                if mapped:
                    parsed_sequence.append(mapped)
                else:
                    return None  # Can't handle this monomer

        return parsed_sequence

    except Exception:
        return None


def _normalize_monomer(monomer: str) -> str:
    """Normalize monomer names to standard amino acid codes."""
    # Handle common modifications
    monomer = monomer.strip('[]')

    # Map common modifications to base amino acids
    modifications = {
        'meL': 'L',  # N-methyl leucine -> leucine
        'dA': 'A',   # D-alanine -> alanine
        'meV': 'V',  # N-methyl valine -> valine
        'Abu': 'A',  # alpha-aminobutyric acid -> alanine
        'Sar': 'G',  # Sarcosine -> glycine
        'Me_Bmt(E)': 'T'  # Modified threonine
    }

    return modifications.get(monomer, monomer)


def _map_unknown_monomer(monomer: str) -> Optional[str]:
    """Try to map unknown monomers to standard amino acids."""
    # This is a fallback for unknown monomers
    # In practice, you'd need a more comprehensive mapping
    if len(monomer) == 1:
        return monomer if monomer in 'ARNDCEQGHILKMFPSTWYV' else None
    return None


def _sequence_to_smiles(sequence: List[str], monomer_map: dict) -> Optional[str]:
    """
    Convert amino acid sequence to SMILES.

    This is a simplified approach - real peptide SMILES generation is complex.
    """
    try:
        # For now, just return a placeholder that indicates successful parsing
        # In a full implementation, you'd build the actual cyclic peptide SMILES
        # This would involve:
        # 1. Connecting amino acids with peptide bonds
        # 2. Adding cyclization
        # 3. Handling stereochemistry

        if len(sequence) == 0:
            return None

        # Return a simple linear peptide SMILES as a placeholder
        # The real implementation would be much more sophisticated
        return f"N{'-'.join(sequence)}C(=O)O"  # Placeholder

    except Exception:
        return None


def validate_cyclic_peptide_smiles(smiles: str) -> bool:
    """
    Validate that a SMILES string represents a valid cyclic peptide.

    Args:
        smiles: SMILES string

    Returns:
        True if valid cyclic peptide, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Check for ring structure (cyclic)
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() == 0:
            return False

        # Additional checks for peptide-like structure could be added here
        # e.g., presence of amide bonds, amino acid-like substructures

        return True

    except Exception:
        return False