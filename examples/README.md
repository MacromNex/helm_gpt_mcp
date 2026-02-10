# HELM-GPT Use Cases Examples

This directory contains standalone Python scripts demonstrating key use cases for cyclic peptide design and analysis using HELM-GPT.

## Quick Start

All scripts require the Python 3.7 environment (`./env_py3.7`) with HELM-GPT dependencies:

```bash
# Activate the environment
mamba activate ./env_py3.7

# Run any script with --help to see options
python examples/use_case_1_helm_to_smiles.py --help
```

## Use Cases Overview

### 1. HELM to SMILES Conversion (`use_case_1_helm_to_smiles.py`)
**Purpose**: Convert cyclic peptide sequences from HELM notation to SMILES chemical representation.

```bash
# Convert sequences from sample dataset
python examples/use_case_1_helm_to_smiles.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --limit 10 \
  --output helm_to_smiles_results.csv

# Convert single HELM sequence
python examples/use_case_1_helm_to_smiles.py \
  --input "PEPTIDE1{A.R.G.D}$PEPTIDE1,PEPTIDE1,1:R1-4:R2$$$" \
  --output single_conversion.csv
```

### 2. Generate Novel Peptides (`use_case_2_generate_peptides.py`)
**Purpose**: Generate novel cyclic peptide sequences using a trained transformer model.

**Prerequisites**: Requires a pre-trained model (train using use_case_5)

```bash
# Generate 100 novel sequences
python examples/use_case_2_generate_peptides.py \
  --model path/to/trained_model.pt \
  --n_samples 100 \
  --temperature 0.8 \
  --output generated_peptides.csv
```

### 3. Predict Cell Permeability (`use_case_3_predict_permeability.py`)
**Purpose**: Predict cell membrane permeability using trained random forest models.

```bash
# Predict permeability for sample sequences
python examples/use_case_3_predict_permeability.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --model examples/data/models/regression_rf.pkl \
  --limit 20 \
  --output permeability_predictions.csv
```

### 4. Predict KRAS Binding (`use_case_4_predict_kras_binding.py`)
**Purpose**: Predict KRAS protein binding affinity using XGBoost models.

```bash
# Predict KRAS binding for sample sequences
python examples/use_case_4_predict_kras_binding.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --model examples/data/models/kras_xgboost_reg.pkl \
  --limit 20 \
  --output kras_binding_predictions.csv
```

### 5. Train Prior Model (`use_case_5_train_prior_model.py`)
**Purpose**: Train a transformer prior model on peptide sequence datasets.

```bash
# Train a model on bioactive peptides
python examples/use_case_5_train_prior_model.py \
  --train_data examples/data/sequences/biotherapeutics_dict_prot_flt.csv \
  --output_dir ./my_trained_model \
  --n_epochs 20 \
  --batch_size 64 \
  --learning_rate 3e-4
```

### 6. Optimize Peptides (`use_case_6_optimize_peptides.py`)
**Purpose**: Train reinforcement learning agent to optimize peptides for specific properties.

**Prerequisites**: Requires a pre-trained prior model

```bash
# Optimize for cell permeability
python examples/use_case_6_optimize_peptides.py \
  --prior ./my_trained_model/gpt_model_final.pt \
  --task permeability \
  --n_steps 1000 \
  --batch_size 32 \
  --output_dir ./optimized_for_permeability

# Optimize for KRAS binding
python examples/use_case_6_optimize_peptides.py \
  --prior ./my_trained_model/gpt_model_final.pt \
  --task kras_kd \
  --n_steps 1000 \
  --output_dir ./optimized_for_kras

# Multi-objective optimization (KRAS + permeability)
python examples/use_case_6_optimize_peptides.py \
  --prior ./my_trained_model/gpt_model_final.pt \
  --task kras_perm \
  --n_steps 1000 \
  --output_dir ./optimized_multi_objective
```

## Data Files

### Sequence Datasets

**`data/sequences/CycPeptMPDB_Peptide_All.csv`**
- Cyclic peptide database with experimental data
- Contains HELM sequences, SMILES, permeability measurements
- ~1,000+ cyclic peptides with property annotations

**`data/sequences/CycPeptMPDB_Monomer_All.csv`**
- Individual amino acid/monomer property database
- Natural and non-natural amino acids
- Molecular descriptors and properties

**`data/sequences/biotherapeutics_dict_prot_flt.csv`**
- ChEMBL bioactive peptides dataset
- 20,784 bioactive peptide sequences in HELM notation
- Suitable for training prior models

### Pre-trained Models

**`data/models/regression_rf.pkl`**
- Random forest model for cell permeability prediction
- Trained on molecular descriptors and Morgan fingerprints
- Predicts PAMPA, Caco2, MDCK, RRCK permeability

**`data/models/kras_xgboost_reg.pkl`**
- XGBoost model for KRAS binding affinity prediction
- Trained on Morgan fingerprints
- Predicts binding KD values (μM)

## Example Workflows

### Property Prediction Pipeline

```bash
# 1. Convert HELM to SMILES (for validation)
python examples/use_case_1_helm_to_smiles.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --limit 50 --output step1_smiles.csv

# 2. Predict permeability
python examples/use_case_3_predict_permeability.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --limit 50 --output step2_permeability.csv

# 3. Predict KRAS binding
python examples/use_case_4_predict_kras_binding.py \
  --input examples/data/sequences/CycPeptMPDB_Peptide_All.csv \
  --limit 50 --output step3_kras.csv
```

### Model Training and Generation Pipeline

```bash
# 1. Train prior model (takes 1-2 hours on CPU)
python examples/use_case_5_train_prior_model.py \
  --train_data examples/data/sequences/biotherapeutics_dict_prot_flt.csv \
  --output_dir ./trained_model \
  --n_epochs 10 \
  --batch_size 64

# 2. Generate novel sequences
python examples/use_case_2_generate_peptides.py \
  --model ./trained_model/gpt_model_final.pt \
  --n_samples 500 \
  --output novel_sequences.csv

# 3. Evaluate generated sequences
python examples/use_case_3_predict_permeability.py \
  --input novel_sequences.csv \
  --helm_column helm_sequence \
  --output novel_permeability.csv
```

### Optimization Pipeline

```bash
# 1. Train prior model
python examples/use_case_5_train_prior_model.py \
  --train_data examples/data/sequences/biotherapeutics_dict_prot_flt.csv \
  --output_dir ./prior_model --n_epochs 10

# 2. Optimize for permeability
python examples/use_case_6_optimize_peptides.py \
  --prior ./prior_model/gpt_model_final.pt \
  --task permeability \
  --n_steps 500 --output_dir ./optimized_perm

# 3. Generate optimized sequences
python examples/use_case_2_generate_peptides.py \
  --model ./optimized_perm/gpt_model_best.pt \
  --n_samples 100 --output optimized_sequences.csv

# 4. Validate optimization
python examples/use_case_3_predict_permeability.py \
  --input optimized_sequences.csv \
  --helm_column helm_sequence \
  --output validation_results.csv
```

## Common Parameters

### Input/Output
- `--input, -i`: Input CSV file or single HELM sequence
- `--output, -o`: Output CSV file name
- `--helm_column`: Column name containing HELM sequences (default: "HELM")
- `--limit, -l`: Limit number of sequences to process

### Model Parameters
- `--model, -m`: Path to trained model (.pt file)
- `--batch_size`: Batch size for processing
- `--device`: Computation device (cpu/cuda/auto)
- `--max_len`: Maximum sequence length

### Training Parameters
- `--n_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimization
- `--n_steps`: Number of RL training steps
- `--temperature`: Sampling temperature (higher = more diverse)

## Error Handling

All scripts include comprehensive error handling:

- **Import errors**: Clear messages about missing dependencies
- **File not found**: Helpful suggestions for file paths
- **Invalid HELM**: Graceful handling of invalid sequences
- **Model loading**: Clear error messages for model issues
- **Memory errors**: Suggestions for reducing batch sizes

## Output Formats

### Conversion Results
- HELM sequence, SMILES, success status, error messages
- Validation statistics and sample outputs

### Predictions
- Sequence ID, HELM sequence, prediction scores
- Interpretations (e.g., "High permeability", "Strong binding")
- Statistics (mean, median, distribution)

### Generated Sequences
- Generated HELM sequences with validity flags
- Length statistics and diversity metrics
- Sample sequences for inspection

### Training Outputs
- Model checkpoints (.pt files)
- Configuration files (.json)
- Training logs and loss curves
- Sample sequences from each training step

## Performance Notes

- **Memory Usage**: Model training requires 2-4GB RAM
- **Training Time**: Prior training takes 30-60 minutes per epoch
- **Generation Speed**: ~100-1000 sequences per minute
- **Prediction Speed**: ~10-100 sequences per second

## Troubleshooting

### Common Issues

**"ImportError: No module named 'torch'"**
```bash
mamba run -p ./env_py3.7 pip install torch==1.13.1+cpu
```

**"Model file not found"**
- Ensure you've trained a model using `use_case_5_train_prior_model.py`
- Check file paths and permissions

**"CUDA out of memory"**
- Use `--device cpu` to force CPU computation
- Reduce `--batch_size` parameter

**"Invalid HELM sequence"**
- Check HELM notation format
- Use `use_case_1_helm_to_smiles.py` to validate sequences

### Getting Help

Each script supports `--help`:
```bash
python examples/use_case_1_helm_to_smiles.py --help
```

For detailed technical information, see:
- `../reports/step3_use_cases.md`: Comprehensive use case documentation
- `../reports/step3_environment.md`: Environment setup details
- `../README.md`: Main project documentation