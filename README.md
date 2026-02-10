# CycPep-Tools MCP

> MCP server for cyclic peptide computational analysis - providing HELM-to-SMILES conversion, permeability prediction, and KRAS binding affinity analysis

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

CycPep-Tools provides computational analysis capabilities for cyclic peptides through both standalone scripts and an MCP (Model Context Protocol) server. This enables seamless integration with AI assistants like Claude Code for interactive analysis of cyclic peptide sequences.

### Features
- **HELM-to-SMILES conversion** for cyclic peptide notation standardization
- **Membrane permeability prediction** using trained Random Forest models
- **KRAS binding affinity prediction** using XGBoost regression models
- **Batch processing capabilities** for virtual screening workflows
- **Dual API support** (synchronous for fast operations, asynchronous for large datasets)
- **Claude Code integration** for AI-assisted cyclic peptide analysis

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Main MCP environment (Python 3.10)
├── env_py3.7/             # Legacy HELM-GPT environment (Python 3.7)
├── src/
│   ├── server.py           # MCP server with 15 tools
│   ├── utils.py            # Shared utilities
│   └── jobs/               # Job management system
├── scripts/
│   ├── helm_to_smiles.py         # HELM notation converter
│   ├── predict_permeability.py   # Membrane permeability predictor
│   ├── predict_kras_binding.py   # KRAS binding affinity predictor
│   └── lib/                      # Shared utilities
├── examples/
│   └── data/               # Demo datasets and models
│       ├── sequences/      # Sample cyclic peptide datasets (23 MB)
│       ├── models/         # Pre-trained ML models (17 MB)
│       └── structures/     # Sample structures
├── configs/                # JSON configuration files
└── repo/                   # Original HELM-GPT repository
```

---

## Installation

### Quick Setup

Run the automated setup script:

```bash
./quick_setup.sh
```

This will create the environment and install all dependencies automatically.

### Manual Setup (Advanced)

For manual installation or customization, follow these steps.

#### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- 8+ GB disk space (6.8 GB for environments + 1.4 GB for data/models)

#### Environment Setup

Please follow the procedure below to set up the environment:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/helm_gpt_mcp

# Determine package manager (prefer mamba over conda)
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
else
    PKG_MGR="conda"
fi
echo "Using package manager: $PKG_MGR"

# Create main MCP environment (Python 3.10)
$PKG_MGR create -p ./env python=3.10 pip -y

# Activate environment
$PKG_MGR activate ./env

# Install MCP dependencies
$PKG_MGR run -p ./env pip install loguru click pandas numpy tqdm
$PKG_MGR run -p ./env pip install --force-reinstall --no-cache-dir fastmcp
$PKG_MGR run -p ./env pip install scikit-learn scipy

# Test installation
$PKG_MGR run -p ./env python -c "import fastmcp; print('FastMCP version:', fastmcp.__version__)"
```

### Legacy Environment (for enhanced HELM processing)

```bash
# Create legacy environment for HELM-GPT compatibility
$PKG_MGR env create -f repo/helm-gpt/environment.yml -p ./env_py3.7

# This may take 10-15 minutes to complete
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example Use Case |
|--------|-------------|------------------|
| `scripts/helm_to_smiles.py` | Convert HELM notation to SMILES | Chemical database standardization |
| `scripts/predict_permeability.py` | Predict membrane permeability | Drug-like property screening |
| `scripts/predict_kras_binding.py` | Predict KRAS binding affinity | Therapeutic target analysis |

### Script Examples

#### Convert HELM to SMILES

```bash
# Activate environment
mamba activate ./env

# Single HELM conversion
python scripts/helm_to_smiles.py \
  --input "PEPTIDE1{G.R.G.D.S.P}\$\$\$\$" \
  --output results/converted.csv

# Batch CSV processing
python scripts/helm_to_smiles.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --output results/batch_conversion.csv \
  --limit 100
```

**Parameters:**
- `--input, -i`: HELM string or CSV file path (required)
- `--output, -o`: Output CSV file path (optional)
- `--helm_column`: Column name containing HELM sequences (default: "HELM")
- `--limit`: Maximum sequences to process (default: 100)
- `--config`: Configuration file (optional)

#### Predict Membrane Permeability

```bash
python scripts/predict_permeability.py \
  --input "PEPTIDE1{G.R.G.D.S.P}\$\$\$\$" \
  --output results/permeability.csv
```

**Expected Output:**
- Permeability score (-10.0 to 1.0 scale)
- Interpretation (High/Medium/Low permeability)
- Success/failure status for each sequence

#### Predict KRAS Binding

```bash
python scripts/predict_kras_binding.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --output results/kras_binding.csv \
  --limit 50
```

**Expected Output:**
- Binding score (0-1 scale)
- KD value in μM
- Binding strength interpretation (Strong/Moderate/Weak/Very Weak)
- Top binders summary

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name cycpep-tools
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "cycpep-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/helm_gpt_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/helm_gpt_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from cycpep-tools? Give me a brief description of each.
```

#### Single Sequence Analysis
```
Convert this HELM notation to SMILES and predict its permeability: PEPTIDE1{G.R.G.D.S.P}$$$$
```

#### Batch Analysis
```
Analyze the file examples/data/sequences/CycPeptMPDB_Peptide_All.csv:
1. Convert first 10 HELM sequences to SMILES
2. Predict their membrane permeability
3. Identify sequences with high permeability potential
```

#### Workflow Example
```
For the peptide PEPTIDE1{A.R.G.D.F.V}$$$$:
1. Validate the HELM notation
2. Convert to SMILES
3. Predict membrane permeability
4. Predict KRAS binding affinity
5. Summarize drug discovery potential
```

#### Background Job Processing
```
Submit a large permeability prediction job for the file examples/data/sequences/CycPeptMPDB_Peptide_All.csv with 1000 sequences. Monitor the job status and show me results when complete.
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/sequences/CycPeptMPDB_Peptide_All.csv` | Main peptide dataset |
| `@examples/data/sequences/biotherapeutics_dict_prot_flt.csv` | ChEMBL bioactive peptides |
| `@configs/predict_permeability_config.json` | Permeability prediction settings |
| `@configs/predict_kras_binding_config.json` | KRAS binding settings |
| `@results/` | Output directory for results |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "cycpep-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/helm_gpt_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/helm_gpt_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Calculate permeability for cyclic peptide PEPTIDE1{G.R.G.D.S.P}$$$$
```

---

## Available Tools

### Synchronous Tools (Fast Operations < 30 seconds)

These tools return results immediately for typical workloads (<1000 sequences):

| Tool | Description | Parameters | Output |
|------|-------------|------------|--------|
| `helm_to_smiles` | Convert HELM notation to SMILES | `helm_input` or `input_file` | Conversion results with success/error status |
| `predict_permeability` | Predict membrane permeability | `helm_input` or `input_file` | Permeability scores and interpretations |
| `predict_kras_binding` | Predict KRAS binding affinity | `helm_input` or `input_file` | KD values and binding categories |
| `validate_helm_notation` | Validate HELM syntax | `helm_input` | Validation status and structure info |
| `get_server_info` | Server capabilities | None | Tool list and server metadata |
| `get_model_info` | ML model information | None | Model status and availability |

### Asynchronous Tools (Large Datasets > 1000 sequences)

These tools submit background jobs for large-scale processing:

| Tool | Description | Use Case | Monitoring Tools |
|------|-------------|----------|------------------|
| `submit_helm_to_smiles_batch` | Large HELM conversion | >1000 sequences | `get_job_status`, `get_job_result` |
| `submit_permeability_batch` | Large permeability prediction | Virtual screening | `get_job_log`, `cancel_job` |
| `submit_kras_binding_batch` | Large KRAS binding prediction | Target screening | `list_jobs` |

### Job Management Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `get_job_status` | Check job progress | Monitor background processing |
| `get_job_result` | Get completed results | Retrieve final outputs |
| `get_job_log` | View execution logs | Debug failures |
| `cancel_job` | Stop running jobs | Resource management |
| `list_jobs` | List all jobs | Project oversight |
| `cleanup_completed_jobs` | Remove old jobs | Storage maintenance |

---

## Examples

### Example 1: Basic Property Analysis

**Goal:** Analyze drug-like properties for a cyclic RGD peptide

**Using Script:**
```bash
# Convert HELM to SMILES
python scripts/helm_to_smiles.py \
  --input "PEPTIDE1{G.R.G.D.S.P}\$\$\$\$" \
  --output results/rgd_peptide.csv

# Predict permeability
python scripts/predict_permeability.py \
  --input "PEPTIDE1{G.R.G.D.S.P}\$\$\$\$" \
  --output results/rgd_permeability.csv
```

**Using MCP (in Claude Code):**
```
Analyze the drug properties of this RGD peptide: PEPTIDE1{G.R.G.D.S.P}$$$$

1. Convert to SMILES format
2. Predict membrane permeability
3. Predict KRAS binding affinity
4. Interpret results for drug discovery
```

**Expected Output:**
- SMILES: Chemical structure representation
- Permeability: Score with High/Medium/Low classification
- KRAS Binding: KD value with binding strength category

### Example 2: Dataset Screening

**Goal:** Screen cyclic peptide library for therapeutic potential

**Using MCP (in Claude Code):**
```
Screen the cyclic peptide database @examples/data/sequences/CycPeptMPDB_Peptide_All.csv:

1. Process first 50 sequences
2. Convert all HELM notations to SMILES
3. Predict membrane permeability for all
4. Identify top 10 peptides with highest permeability
5. For the top candidates, predict KRAS binding
6. Create a summary table of the best drug candidates
```

**Expected Process:**
1. HELM validation and conversion (~30 seconds)
2. Permeability screening (~45 seconds)
3. KRAS binding analysis for top candidates (~20 seconds)
4. Summary report with ranked candidates

### Example 3: Large-Scale Virtual Screening

**Goal:** Screen large peptide library (>1000 sequences) for KRAS inhibitors

**Using MCP (in Claude Code):**
```
Perform large-scale screening of @examples/data/sequences/biotherapeutics_dict_prot_flt.csv:

1. Submit background job for KRAS binding prediction (limit 1000)
2. Monitor job progress
3. When complete, identify top 20 KRAS binders
4. For top binders, also predict permeability
5. Create final drug candidate ranking
```

**Expected Workflow:**
1. Job submission (immediate)
2. Background processing (5-10 minutes for 1000 sequences)
3. Results analysis and ranking
4. Final candidate selection with dual criteria

---

## Demo Data

The `examples/data/` directory contains datasets for testing and learning:

| Dataset | Size | Description | Use With |
|---------|------|-------------|----------|
| `sequences/CycPeptMPDB_Peptide_All.csv` | 15.8 MB, ~50K sequences | Cyclic peptide database with HELM notation | All tools |
| `sequences/biotherapeutics_dict_prot_flt.csv` | 6.7 MB, ~20K sequences | ChEMBL bioactive peptides | Training and screening |
| `sequences/CycPeptMPDB_Monomer_All.csv` | 724 KB | Amino acid property database | Reference data |
| `models/regression_rf.pkl` | 14.8 MB | Random Forest permeability model | Permeability prediction |
| `models/kras_xgboost_reg.pkl` | 3.1 MB | XGBoost KRAS binding model | KRAS binding prediction |

**Sample HELM Sequences:**
```
PEPTIDE1{[Abu].[Sar].[meL].V.[meL].A.[dA].[meL].[meL].[meV].[Me_Bmt(E)]}$PEPTIDE1,PEPTIDE1,1:R3-20:R3$$$
PEPTIDE1{G.R.G.D.S.P}$$$$
PEPTIDE1{R.G.D.F.V}$$$$
```

---

## Configuration Files

The `configs/` directory contains settings for customizing tool behavior:

### Global Configuration (`configs/default_config.json`)
```json
{
  "environment": "production",
  "verbosity": 1,
  "paths": {
    "models_base": "examples/data/models",
    "scripts_base": "scripts"
  },
  "repo_fallback": true,
  "max_limit": 10000
}
```

### Permeability Settings (`configs/predict_permeability_config.json`)
```json
{
  "model_path": "examples/data/models/regression_rf.pkl",
  "model_type": "random_forest",
  "score_thresholds": {
    "high": 0.7,
    "medium": 0.4,
    "low": 0.0
  },
  "include_statistics": true
}
```

### KRAS Binding Settings (`configs/predict_kras_binding_config.json`)
```json
{
  "model_path": "examples/data/models/kras_xgboost_reg.pkl",
  "binding_thresholds": {
    "strong": 1.0,
    "moderate": 10.0,
    "weak": 100.0
  },
  "top_binders_count": 5
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
mamba run -p ./env pip install loguru click pandas numpy tqdm fastmcp scikit-learn scipy
```

**Problem:** FastMCP import errors
```bash
# Force reinstall FastMCP
mamba run -p ./env pip install --force-reinstall --no-cache-dir fastmcp

# Test installation
mamba run -p ./env python -c "import fastmcp; print('FastMCP version:', fastmcp.__version__)"
```

**Problem:** Model loading failures
```bash
# Check model files exist
ls -la examples/data/models/
# Should show:
# regression_rf.pkl (14.8 MB)
# kras_xgboost_reg.pkl (3.1 MB)

# If missing, extract from original repository
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove cycpep-tools
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Invalid HELM notation errors
```
Ensure HELM strings use proper peptide notation. For cyclic peptides:
- Linear: PEPTIDE1{G.R.G.D.S.P}$$$$
- With connections: PEPTIDE1{A.G.C}$PEPTIDE1,PEPTIDE1,1:R3-3:R3$$$
```

**Problem:** Tools not responding
```bash
# Test server directly
mamba run -p ./env python -c "
from src.server import mcp
print('Server name:', mcp.name)
"
```

### Performance Issues

**Problem:** Slow processing
- **Solution:** Reduce batch size using `limit` parameter
- **Use async tools:** For datasets >1000 sequences, use `submit_*_batch` tools
- **Check memory:** Ensure 4+ GB available RAM for large batches

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job logs
python -c "from src.server import get_job_log; print(get_job_log('your_job_id'))"
```

### Data Issues

**Problem:** CSV parsing errors
- **Check format:** Ensure CSV has proper headers and HELM column
- **Check encoding:** Use UTF-8 encoding for CSV files
- **Validate HELM:** Use `validate_helm_notation` tool first

**Expected CSV Format:**
```csv
HELM,Name,Description
"PEPTIDE1{G.R.G.D.S.P}$$$$","RGD-peptide","Integrin binder"
```

---

## Performance Characteristics

### Processing Speed
- **HELM to SMILES**: ~33 sequences/second
- **Permeability prediction**: ~33 sequences/second
- **KRAS binding**: ~33 sequences/second
- **Combined analysis**: ~15 sequences/second

### Resource Usage
- **Base server**: ~50 MB RAM
- **With models loaded**: ~100 MB RAM
- **Per 1000 sequences**: +20 MB RAM
- **Disk space**: 8+ GB total (environments + data)

### Timeout Recommendations
- **Synchronous tools**: 30 seconds (up to 1000 sequences)
- **Asynchronous jobs**: No timeout (background execution)
- **Job monitoring**: Check every 30-60 seconds

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Run basic tests
python tests/test_simple.py

# Run server startup tests
python tests/test_server_start.py

# Run integration tests
python tests/run_integration_tests.py
```

### Starting Development Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py

# Access MCP Inspector at http://localhost:6274/
```

---

## License

This project is based on the HELM-GPT framework for cyclic peptide design and analysis.

## Credits

Based on [HELM-GPT: De novo Design of Bioactive Helical Peptides with Generative Pre-trained Transformers](https://github.com/wenhao-gao/helm-gp) by Wenhao Gao et al.

### Model Attribution
- **Permeability Model**: Random Forest trained on cyclic peptide permeability data
- **KRAS Binding Model**: XGBoost trained on KRAS inhibition data
- **HELM Processing**: Based on HELM notation standard and utilities

---

## Getting Help

For issues and questions:
- Check the troubleshooting section above
- Review example prompts for proper usage
- Test with smaller datasets before large-scale processing
- Ensure proper environment setup and model file availability

**Common Commands:**
```bash
# Check server status
claude mcp list

# Test basic functionality
python scripts/helm_to_smiles.py --input "PEPTIDE1{G.R.G.D.S.P}\$\$\$\$"

# View available tools
python -c "from src.server import mcp; print('Tools available')"
```