# Step 3: Use Cases Report

## Scan Information
- **Scan Date**: 2025-12-31
- **Filter Applied**: macrocyclic peptide design using HELM-GPT, HELM notation generation, peptide optimization
- **Python Version**: 3.7 (legacy) + 3.10 (MCP)
- **Environment Strategy**: Dual environment (./env for MCP, ./env_py3.7 for HELM-GPT)

## Use Cases Extracted

### UC-001: HELM to SMILES Conversion
- **Description**: Convert cyclic peptide sequences from HELM notation to SMILES chemical representation
- **Script Path**: `examples/use_case_1_helm_to_smiles.py`
- **Complexity**: Simple
- **Priority**: High
- **Environment**: `./env_py3.7` (requires HELM-GPT utils)
- **Source**: `utils/helm_utils.py` functions, repository analysis

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input | file/string | CSV file with HELM sequences or single HELM string | --input, -i |
| helm_column | string | Column name containing HELM sequences | --helm_column |
| limit | integer | Number of sequences to process | --limit, -l |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| output | file | CSV with HELM sequences, SMILES, success status, errors |

**Example Usage:**
```bash
python examples/use_case_1_helm_to_smiles.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --output helm_to_smiles_results.csv \
  --limit 50
```

**Example Data**: `examples/data/sequences/CycPeptMPDB_Peptide_All.csv`

---

### UC-002: Generate Novel Peptides
- **Description**: Generate novel cyclic peptide sequences using a trained HELM-GPT transformer model
- **Script Path**: `examples/use_case_2_generate_peptides.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env_py3.7` (requires PyTorch + HELM-GPT)
- **Source**: `generate.py`, `model/model.py`, `model/sampler.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| model | file | Trained GPT model checkpoint (.pt file) | --model, -m |
| n_samples | integer | Number of sequences to generate | --n_samples, -n |
| max_len | integer | Maximum sequence length | --max_len |
| temperature | float | Sampling temperature (diversity control) | --temperature, -t |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| output | file | CSV with generated sequences, validity, length stats |

**Example Usage:**
```bash
python examples/use_case_2_generate_peptides.py \
  --model path/to/trained_model.pt \
  --n_samples 1000 \
  --temperature 0.8 \
  --output generated_peptides.csv
```

**Example Data**: Requires pre-trained model (not included, needs training)

---

### UC-003: Predict Cell Permeability
- **Description**: Predict cell membrane permeability for cyclic peptides using random forest models
- **Script Path**: `examples/use_case_3_predict_permeability.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env_py3.7` (requires RDKit + scikit-learn)
- **Source**: `agent/scoring/permeability.py`, random forest model

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input | file/string | CSV with HELM sequences or single HELM string | --input, -i |
| helm_column | string | Column name containing HELM sequences | --helm_column |
| model | file | Custom permeability model (.pkl) | --model |
| limit | integer | Number of sequences to process | --limit, -l |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| output | file | CSV with permeability scores and interpretations |

**Example Usage:**
```bash
python examples/use_case_3_predict_permeability.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --model examples/data/models/regression_rf.pkl \
  --output permeability_predictions.csv
```

**Example Data**:
- `examples/data/sequences/CycPeptMPDB_Peptide_All.csv`
- `examples/data/models/regression_rf.pkl`

---

### UC-004: Predict KRAS Binding Affinity
- **Description**: Predict KRAS protein binding affinity (KD values) for cyclic peptides using XGBoost models
- **Script Path**: `examples/use_case_4_predict_kras_binding.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env_py3.7` (requires XGBoost + RDKit)
- **Source**: `agent/scoring/kras.py`, XGBoost regression model

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| input | file/string | CSV with HELM sequences or single HELM string | --input, -i |
| helm_column | string | Column name containing HELM sequences | --helm_column |
| model | file | Custom KRAS binding model (.pkl) | --model |
| limit | integer | Number of sequences to process | --limit, -l |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| output | file | CSV with binding scores, KD values, binding strength interpretation |

**Example Usage:**
```bash
python examples/use_case_4_predict_kras_binding.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --model examples/data/models/kras_xgboost_reg.pkl \
  --output kras_binding_predictions.csv
```

**Example Data**:
- `examples/data/sequences/CycPeptMPDB_Peptide_All.csv`
- `examples/data/models/kras_xgboost_reg.pkl`

---

### UC-005: Train Prior Model
- **Description**: Train a transformer prior model on cyclic peptide sequence datasets for generative modeling
- **Script Path**: `examples/use_case_5_train_prior_model.py`
- **Complexity**: Complex
- **Priority**: Medium
- **Environment**: `./env_py3.7` (requires PyTorch + HELM-GPT)
- **Source**: `train_prior.py`, `prior/trainer.py`, `model/model.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| train_data | file | Training dataset CSV with HELM sequences | --train_data |
| valid_data | file | Validation dataset CSV (optional) | --valid_data |
| helm_column | string | Column containing HELM sequences | --helm_column |
| n_epochs | integer | Number of training epochs | --n_epochs |
| batch_size | integer | Batch size | --batch_size |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| output_dir | directory | Directory with trained model checkpoints and configs |

**Example Usage:**
```bash
python examples/use_case_5_train_prior_model.py \
  --train_data examples/data/sequences/biotherapeutics_dict_prot_flt.csv \
  --output_dir ./my_trained_model \
  --n_epochs 50 \
  --batch_size 128
```

**Example Data**: `examples/data/sequences/biotherapeutics_dict_prot_flt.csv`

---

### UC-006: Optimize Peptides with Agent Training
- **Description**: Train reinforcement learning agent to optimize cyclic peptides for specific properties
- **Script Path**: `examples/use_case_6_optimize_peptides.py`
- **Complexity**: Complex
- **Priority**: High
- **Environment**: `./env_py3.7` (requires PyTorch + HELM-GPT + property models)
- **Source**: `train_agent.py`, `agent/agent_trainer.py`, `agent/scoring_functions.py`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| prior | file | Pre-trained prior model (.pt file) | --prior |
| task | choice | Optimization target (permeability/kras_kd/kras_perm) | --task |
| n_steps | integer | Number of training steps | --n_steps |
| batch_size | integer | Batch size | --batch_size |
| sigma | float | Augmented likelihood parameter | --sigma |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| output_dir | directory | Optimized agent model and sample sequences |

**Example Usage:**
```bash
python examples/use_case_6_optimize_peptides.py \
  --prior ./my_trained_model/gpt_model_final.pt \
  --task permeability \
  --n_steps 1000 \
  --output_dir ./optimized_for_permeability
```

**Example Data**: Requires pre-trained prior model from UC-005

---

## Summary

| Metric | Count |
|--------|-------|
| Total Use Cases Found | 6 |
| Scripts Created | 6 |
| High Priority | 4 |
| Medium Priority | 1 |
| Low Priority | 0 |
| Simple Complexity | 1 |
| Medium Complexity | 3 |
| Complex Complexity | 2 |
| Demo Data Copied | Yes |

## Priority Distribution

**High Priority (4 use cases):**
- UC-001: HELM to SMILES Conversion (core utility)
- UC-002: Generate Novel Peptides (main generative function)
- UC-003: Predict Cell Permeability (key property prediction)
- UC-006: Optimize Peptides (main optimization workflow)

**Medium Priority (2 use cases):**
- UC-004: Predict KRAS Binding (specialized property prediction)
- UC-005: Train Prior Model (advanced model training)

## Demo Data Index

| Source | Destination | Description |
|--------|-------------|-------------|
| `repo/helm-gpt/data/prior/CycPeptMPDB/CycPeptMPDB_Peptide_All.csv` | `examples/data/sequences/CycPeptMPDB_Peptide_All.csv` | Cyclic peptide database with HELM, SMILES, permeability data |
| `repo/helm-gpt/data/prior/CycPeptMPDB/CycPeptMPDB_Monomer_All.csv` | `examples/data/sequences/CycPeptMPDB_Monomer_All.csv` | Monomer/amino acid property database |
| `repo/helm-gpt/data/prior/chembl32/biotherapeutics_dict_prot_flt.csv` | `examples/data/sequences/biotherapeutics_dict_prot_flt.csv` | ChEMBL bioactive peptides (20,784 sequences) |
| `repo/helm-gpt/data/cpp/regression_rf.pkl` | `examples/data/models/regression_rf.pkl` | Trained random forest permeability model |
| `repo/helm-gpt/data/kras_kd/kras_xgboost_reg.pkl` | `examples/data/models/kras_xgboost_reg.pkl` | Trained XGBoost KRAS binding model |

## Workflow Dependencies

```
UC-005 (Train Prior) → UC-002 (Generate Peptides)
                    ↘ UC-006 (Optimize Peptides)

UC-001 (HELM→SMILES) → UC-003 (Predict Permeability)
                    → UC-004 (Predict KRAS)
                    ↗ UC-006 (Optimize Peptides)
```

## Technical Requirements

**All Use Cases Require:**
- Python 3.7 environment (`./env_py3.7`)
- HELM-GPT library and utilities
- Basic Python packages (pandas, numpy)

**Specific Requirements:**
- **UC-001**: HELM utilities, RDKit (molecular conversion)
- **UC-002, UC-005, UC-006**: PyTorch, HELM-GPT model classes
- **UC-003**: RDKit, scikit-learn (random forest)
- **UC-004**: RDKit, XGBoost
- **UC-006**: All property prediction models

## Error Handling

All scripts include:
- Import error detection with helpful messages
- Environment verification
- Input validation
- Graceful failure handling
- Progress reporting
- Results statistics and interpretation

## Example Command Workflows

**Quick Start (property prediction):**
```bash
# 1. Convert HELM to SMILES
python examples/use_case_1_helm_to_smiles.py --input "PEPTIDE1{A.R.G}$$$"

# 2. Predict permeability
python examples/use_case_3_predict_permeability.py --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv --limit 10

# 3. Predict KRAS binding
python examples/use_case_4_predict_kras_binding.py --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv --limit 10
```

**Advanced Workflow (model training and optimization):**
```bash
# 1. Train prior model
python examples/use_case_5_train_prior_model.py \
  --train_data examples/data/sequences/biotherapeutics_dict_prot_flt.csv \
  --output_dir ./my_model --n_epochs 20

# 2. Generate novel peptides
python examples/use_case_2_generate_peptides.py \
  --model ./my_model/gpt_model_final.pt --n_samples 500

# 3. Optimize for permeability
python examples/use_case_6_optimize_peptides.py \
  --prior ./my_model/gpt_model_final.pt --task permeability \
  --n_steps 500 --output_dir ./optimized_peptides
```