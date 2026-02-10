#!/usr/bin/env python3
"""
Mock Scoring Server for Testing

A simple Flask server that returns placeholder scores for testing the
Boltz2 and Rosetta MCP client infrastructure without requiring actual
computational backends.

Usage:
    python scripts/mock_scoring_server.py [--port PORT]

    Default port: 8001 (serves both Rosetta and Boltz2 endpoints)

Endpoints:
    POST /rosetta/score - Mock Rosetta scoring
    POST /biology/mit/boltz2/predict - Mock Boltz2 structure prediction
    GET /health - Health check
"""

import argparse
import json
import random
from datetime import datetime

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Flask is required for the mock server. Install with: pip install flask")
    exit(1)

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'server': 'mock_scoring_server',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
        'endpoints': [
            '/rosetta/score',
            '/biology/mit/boltz2/predict',
            '/health'
        ]
    })


@app.route('/rosetta/score', methods=['POST', 'OPTIONS'])
def mock_rosetta_score():
    """
    Mock Rosetta scoring endpoint.

    Returns realistic mock values for:
    - ddG: Binding free energy (-30 to +5 kcal/mol)
    - SAP: Spatial Aggregation Propensity (0 to 150)
    - CMS: Contact Map Score (0 to 1)
    - total_score: Rosetta total energy (-500 to +100)
    - interface_score: Interface energy (-100 to +20)
    """
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json or {}
        sequence = data.get('sequence', '')
        requested_scores = data.get('requested_scores', ['ddG', 'SAP', 'CMS'])

        # Generate mock scores based on sequence properties
        seq_len = len(sequence)

        # Realistic score ranges based on peptide scoring literature
        scores = {}

        if 'ddG' in requested_scores or not requested_scores:
            # ddG typically ranges from -30 (very strong binding) to +5 (no binding)
            # Shorter peptides often have weaker binding
            base_ddg = -15 + random.gauss(0, 5)
            if seq_len < 10:
                base_ddg += 5  # Penalty for short peptides
            scores['ddG'] = round(max(-30, min(5, base_ddg)), 2)

        if 'SAP' in requested_scores or not requested_scores:
            # SAP ranges from 0 (no aggregation) to 150+ (high aggregation)
            # Hydrophobic-rich sequences have higher SAP
            scores['SAP'] = round(random.uniform(10, 80), 2)

        if 'CMS' in requested_scores or not requested_scores:
            # CMS ranges from 0 to 1 (higher = better contact)
            scores['CMS'] = round(random.uniform(0.4, 0.95), 3)

        if 'total_score' in requested_scores:
            # Rosetta total score, more negative = more stable
            scores['total_score'] = round(-200 + random.gauss(0, 50), 2)

        if 'interface_score' in requested_scores:
            # Interface score, more negative = better interface
            scores['interface_score'] = round(-30 + random.gauss(0, 15), 2)

        # Generate a mock PDB structure (minimal placeholder)
        mock_pdb = _generate_mock_pdb(sequence) if data.get('return_structure') else None

        response = {
            'status': 'success',
            'scores': scores,
            'sequence': sequence,
            'sequence_length': seq_len,
            'computation_time_ms': random.randint(100, 500),
            'server': 'mock_rosetta'
        }

        if mock_pdb:
            response['pdb'] = mock_pdb

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400


@app.route('/biology/mit/boltz2/predict', methods=['POST', 'OPTIONS'])
def mock_boltz2_predict():
    """
    Mock Boltz2 structure prediction endpoint.

    Returns realistic mock values for:
    - iptm_scores: Interface pTM (0 to 1)
    - ipae_scores: Interface PAE (0 to 30)
    - affinity_pic50: Predicted pIC50 (3 to 10)
    - complex_ipde_scores: Complex iPDE (0 to 20)
    """
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.json or {}
        polymers = data.get('polymers', [])

        # Extract sequence from polymers
        sequence = ''
        if polymers:
            sequence = polymers[0].get('sequence', '')

        seq_len = len(sequence)

        # Generate mock scores
        # ipTM typically ranges 0.3-0.9 for reasonable predictions
        iptm = round(random.uniform(0.4, 0.85), 3)

        # iPAE typically ranges 0-30 (lower is better)
        ipae = round(random.uniform(2, 15), 2)

        # pIC50 typically ranges 3-10 (higher = stronger binding)
        # pIC50 = -log10(IC50 in M), so 6 = 1 μM, 9 = 1 nM
        affinity_pic50 = round(random.uniform(4, 8), 2)

        # iPDE ranges 0-20 (lower is better)
        ipde = round(random.uniform(1, 10), 2)

        # Generate mock structure (mmCIF format placeholder)
        mock_structure = _generate_mock_mmcif(sequence) if data.get('return_structure') else None

        response = {
            'status': 'success',
            'iptm_scores': [iptm],
            'ipae_scores': [ipae],
            'complex_ipde_scores': [ipde],
            'sequence_length': seq_len,
            'computation_time_ms': random.randint(500, 2000),
            'server': 'mock_boltz2'
        }

        # Add affinity if substrate was provided
        if data.get('affinity'):
            binder_id = data['affinity'].get('binder', 'S')
            response['affinities'] = {
                binder_id: {
                    'affinity_pic50': [affinity_pic50]
                }
            }

        # Add structures
        if mock_structure:
            response['structures'] = [{
                'format': 'mmcif',
                'structure': mock_structure
            }]
        else:
            response['structures'] = []

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400


def _generate_mock_pdb(sequence: str) -> str:
    """Generate a minimal mock PDB structure."""
    if not sequence:
        sequence = "ALA"

    lines = [
        "HEADER    MOCK STRUCTURE",
        f"TITLE     Mock PDB for sequence length {len(sequence)}",
        "MODEL     1"
    ]

    # Add minimal CA atoms
    for i, aa in enumerate(sequence[:20]):  # Limit to first 20
        x = i * 3.8
        y = 0.0
        z = 0.0
        lines.append(
            f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )

    lines.extend([
        "ENDMDL",
        "END"
    ])

    return '\n'.join(lines)


def _generate_mock_mmcif(sequence: str) -> str:
    """Generate a minimal mock mmCIF structure."""
    if not sequence:
        sequence = "A"

    return f"""data_mock_structure
_entry.id mock
_struct.title 'Mock structure for sequence length {len(sequence)}'
_struct.pdbx_descriptor 'Mock peptide'
#
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
1 C CA ALA A 1 0.000 0.000 0.000
#
"""


def main():
    parser = argparse.ArgumentParser(
        description='Mock scoring server for testing Boltz2/Rosetta MCP tools'
    )
    parser.add_argument('--port', type=int, default=8001,
                        help='Port to run server on (default: 8001)')
    parser.add_argument('--host', default='localhost',
                        help='Host to bind to (default: localhost)')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')

    args = parser.parse_args()

    print(f"\n🧪 Starting Mock Scoring Server")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"\n   Endpoints:")
    print(f"   - POST http://{args.host}:{args.port}/rosetta/score")
    print(f"   - POST http://{args.host}:{args.port}/biology/mit/boltz2/predict")
    print(f"   - GET  http://{args.host}:{args.port}/health")
    print(f"\n   Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
