# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HELM-GPT MCP is a FastMCP server (`src/server.py`) for cyclic peptide optimization. It wraps standalone Python scripts as MCP tools, providing HELM-to-SMILES conversion, membrane permeability prediction (Random Forest), and KRAS binding affinity prediction (XGBoost). It also supports RL-based peptide optimization and external server scoring (Boltz2/Rosetta).

## Environment Setup

```bash
# Automated setup (creates ./env with Python 3.10)
./quick_setup.sh

# Manual activation
mamba activate ./env
# or
source ./env/bin/activate
```

Two environments exist:
- `./env/` (Python 3.10): Main MCP server with fastmcp, pandas, scikit-learn, scipy
- `./env_py3.7/` (Python 3.7, optional): Legacy HELM-GPT with RDKit, PyTorch for full optimization

## Running the Server

```bash
# Production
./env/bin/python src/server.py

# Development with MCP Inspector (http://localhost:6274)
fastmcp dev src/server.py

# Register with Claude Code
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

## Running Tests

Tests are standalone Python scripts (no pytest/unittest framework):

```bash
# Basic component tests
python tests/test_simple.py

# Server startup verification
python tests/test_server_start.py

# Full integration tests
python tests/run_integration_tests.py
```

## Running Scripts Directly

```bash
# HELM to SMILES conversion
python scripts/helm_to_smiles.py --input "PEPTIDE1{G.R.G.D.S.P}\$\$\$\$" --output results/out.csv

# Permeability prediction
python scripts/predict_permeability.py --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv --output results/perm.csv --limit 100

# KRAS binding prediction
python scripts/predict_kras_binding.py --input "PEPTIDE1{G.R.G.D.S.P}\$\$\$\$" --output results/kras.csv
```

Common script flags: `--input/-i`, `--output/-o`, `--helm_column` (default: "HELM"), `--limit`, `--config`.

## Architecture

```
MCP Server (src/server.py)        ← FastMCP "cycpep-tools", 21 @mcp.tool() functions
  ├─ src/utils.py                 ← load_config(), format_error_response(), format_success_response()
  ├─ src/jobs/manager.py          ← JobManager: subprocess-based async job execution with threading
  ├─ src/jobs/store.py            ← JobStore: SQLite persistence (jobs/jobs.db)
  │
  ├─ scripts/helm_to_smiles.py    ← run_helm_to_smiles() — imported directly by server
  ├─ scripts/predict_permeability.py ← run_predict_permeability()
  ├─ scripts/predict_kras_binding.py ← run_predict_kras_binding()
  ├─ scripts/optimize_peptides.py    ← run_optimize_peptides() (RL training)
  ├─ scripts/score_with_server.py    ← run_server_scoring() (Boltz2/Rosetta)
  ├─ scripts/mock_scoring_server.py  ← Mock Boltz2/Rosetta server for testing
  └─ scripts/lib/                    ← Shared utilities (helm_utils, prediction_utils, etc.)
```

**Key patterns**:
- Synchronous MCP tools import script functions directly and call them in-process.
- Async/batch tools use `JobManager.submit_job()` which spawns a subprocess running the script with CLI args, logging to `jobs/<job_id>/job.log` and saving output to `jobs/<job_id>/output.json`.
- **Dual-interpreter execution**: `server.py` defines `PY37_PATH` pointing to `env_py3.7/bin/python`. Optimization jobs (`submit_optimize_peptides`) and server scoring jobs (`submit_server_scoring`) pass `python_path=PY37_PATH` to `JobManager.submit_job()`, running them under the Python 3.7 environment which has RDKit/PyTorch. All other jobs use `sys.executable` (the Python 3.10 MCP environment).

## Tool Categories in server.py

- **Synchronous** (< 30s, ≤1000 sequences): `helm_to_smiles`, `predict_permeability`, `predict_kras_binding`, `validate_helm_notation`, `score_with_server`
- **Async batch** (background subprocess): `submit_helm_to_smiles_batch`, `submit_permeability_batch`, `submit_kras_binding_batch`, `submit_server_scoring`
- **Job management**: `get_job_status`, `get_job_result`, `get_job_log`, `cancel_job`, `list_jobs`, `cleanup_completed_jobs`
- **Optimization**: `submit_optimize_peptides`, `check_optimization_requirements`, `get_optimization_tasks`
- **Info**: `get_server_info`, `get_model_info`, `get_server_scorer_info`

## Key Data Paths

- ML models: `examples/data/models/regression_rf.pkl` (permeability), `examples/data/models/kras_xgboost_reg.pkl` (KRAS)
- Demo sequences: `examples/data/sequences/CycPeptMPDB_Peptide_All.csv` (~50K), `examples/data/sequences/biotherapeutics_dict_prot_flt.csv` (~20K)
- Configs: `configs/*.json` — each script has its own config (thresholds, model paths, limits)
- Job data: `jobs/<job_id>/` (metadata.json, job.log, output.json)
- HELM-GPT repo: `repo/helm-gpt/` (submodule for full training/generation)

## Configuration

All tool behavior is driven by JSON configs in `configs/`:
- `default_config.json` — global defaults (limits, paths, logging)
- `helm_to_smiles_config.json` — conversion settings
- `predict_permeability_config.json` — model path, score thresholds (high: 0.7, medium: 0.4)
- `predict_kras_binding_config.json` — model path, binding thresholds (strong: 1.0 μM, moderate: 10.0 μM, weak: 100.0 μM)
- `server_scoring_config.json` — Boltz2/Rosetta server endpoints, timeouts, score definitions

## Important Conventions

- All tool functions return dicts with `"status": "success"|"error"` via `format_success_response()`/`format_error_response()`.
- Script functions (e.g., `run_helm_to_smiles()`) accept both a single HELM string or a CSV file path via the same `input_file` parameter — the function detects which one it received.
- The server adds `SCRIPTS_DIR` to `sys.path` so script modules are imported directly (`from helm_to_smiles import run_helm_to_smiles`).
- HELM notation uses `$$$$` as section delimiters: `PEPTIDE1{residues}$connections$groups$annotations`.
