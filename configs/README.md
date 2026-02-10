# Configuration Files Documentation

Configuration files for cyclic peptide MCP scripts in JSON format.

## Configuration Files

| File | Purpose | Script |
|------|---------|--------|
| `default_config.json` | Global defaults for all scripts | All |
| `helm_to_smiles_config.json` | HELM to SMILES conversion settings | `helm_to_smiles.py` |
| `predict_permeability_config.json` | Permeability prediction settings | `predict_permeability.py` |
| `predict_kras_binding_config.json` | KRAS binding prediction settings | `predict_kras_binding.py` |

## Usage

### Command Line
```bash
python scripts/script_name.py --config configs/config_file.json --input input.csv --output output.csv
```

### Programmatic
```python
import json
from scripts.script_name import run_function

# Load config
with open('configs/config_file.json') as f:
    config = json.load(f)

# Run with config
result = run_function(input_file="data.csv", config=config)
```

### CLI Override
CLI arguments override config file settings:
```bash
python scripts/predict_permeability.py \
  --config configs/predict_permeability_config.json \
  --limit 50 \  # Overrides config limit
  --helm_column "Sequence"  # Overrides config helm_column
```

## Configuration Structure

### Global Settings (default_config.json)

```json
{
  "global": {
    "environment": "auto",         # Environment detection
    "verbosity": "info",          # Logging level
    "error_handling": "graceful"  # Error handling mode
  },
  "input": {
    "default_format": "csv",      # Default input format
    "default_helm_column": "HELM", # Default column name
    "max_file_size_mb": 100       # Max input file size
  },
  "processing": {
    "default_limit": 100,         # Default sequence limit
    "max_limit": 10000,          # Maximum allowed limit
    "timeout_seconds": 300       # Processing timeout
  },
  "repo": {
    "path": "repo/helm-gpt",      # Path to HELM-GPT repo
    "fallback_on_failure": true, # Use fallbacks if repo unavailable
    "lazy_loading": true         # Lazy load repo functions
  },
  "models": {
    "base_path": "examples/data/models", # Model files directory
    "cache_loaded_models": true          # Cache loaded models
  },
  "output": {
    "default_format": "csv",     # Default output format
    "create_directories": true,   # Auto-create output directories
    "include_timestamps": true    # Include timestamps in output
  }
}
```

### Script-Specific Settings

#### HELM to SMILES (helm_to_smiles_config.json)

```json
{
  "input": {
    "format": "csv",              # Input format
    "helm_column": "HELM",        # Column containing HELM sequences
    "encoding": "utf-8"           # File encoding
  },
  "processing": {
    "limit": 100,                 # Max sequences to process
    "validate_input": true,       # Validate HELM sequences
    "skip_empty": true,          # Skip empty sequences
    "error_handling": "continue"  # Continue on individual errors
  },
  "output": {
    "format": "csv",              # Output format
    "include_errors": true,       # Include error information
    "include_sequence_ids": true, # Include sequence IDs
    "validate_output": true       # Validate SMILES output
  },
  "helm": {
    "validation_enabled": true,   # Enable HELM validation
    "conversion_timeout": 30,     # Timeout per conversion (seconds)
    "fallback_on_repo_failure": true # Use fallback if repo fails
  }
}
```

#### Permeability Prediction (predict_permeability_config.json)

```json
{
  "input": {
    "format": "csv",              # Input format
    "helm_column": "HELM",        # Column containing HELM sequences
    "encoding": "utf-8"           # File encoding
  },
  "model": {
    "path": "examples/data/models/regression_rf.pkl", # Model file path
    "type": "random_forest",      # Model type
    "fallback_enabled": true,     # Enable fallback predictions
    "batch_size": 1000           # Batch size for predictions
  },
  "processing": {
    "limit": 100,                 # Max sequences to process
    "validate_helm": true,        # Validate HELM sequences
    "skip_invalid": true,         # Skip invalid sequences
    "conversion_timeout": 30      # HELM conversion timeout
  },
  "prediction": {
    "score_threshold": {
      "high": 0.7,               # High permeability threshold
      "medium": 0.4,             # Medium permeability threshold
      "low": 0.0                 # Low permeability threshold
    },
    "interpretation_enabled": true, # Enable score interpretation
    "confidence_calculation": false # Calculate confidence scores
  },
  "output": {
    "format": "csv",              # Output format
    "include_statistics": true,   # Include statistical summary
    "include_interpretation": true, # Include score interpretation
    "include_metadata": true      # Include execution metadata
  },
  "performance": {
    "timeout_per_sequence": 10,   # Timeout per sequence (seconds)
    "max_retries": 2,            # Max retries for failed predictions
    "parallel_processing": false  # Enable parallel processing
  }
}
```

#### KRAS Binding (predict_kras_binding_config.json)

```json
{
  "input": {
    "format": "csv",              # Input format
    "helm_column": "HELM",        # Column containing HELM sequences
    "encoding": "utf-8"           # File encoding
  },
  "model": {
    "path": "examples/data/models/kras_xgboost_reg.pkl", # Model file path
    "type": "xgboost_regressor",  # Model type
    "fallback_enabled": true,     # Enable fallback predictions
    "score_to_kd_conversion": "log10" # Score to KD conversion method
  },
  "processing": {
    "limit": 100,                 # Max sequences to process
    "validate_helm": true,        # Validate HELM sequences
    "skip_invalid": true,         # Skip invalid sequences
    "conversion_timeout": 30      # HELM conversion timeout
  },
  "prediction": {
    "binding_thresholds": {
      "strong": 1.0,             # Strong binding threshold (μM)
      "moderate": 10.0,          # Moderate binding threshold (μM)
      "weak": 100.0              # Weak binding threshold (μM)
    },
    "score_thresholds": {
      "high": 0.8,               # High score threshold
      "medium": 0.5,             # Medium score threshold
      "low": 0.0                 # Low score threshold
    },
    "units": "μM",               # KD units
    "interpretation_enabled": true # Enable binding interpretation
  },
  "output": {
    "format": "csv",              # Output format
    "include_statistics": true,   # Include statistical summary
    "include_interpretation": true, # Include binding interpretation
    "include_top_binders": true,  # Include top binders list
    "top_binders_count": 5,      # Number of top binders to show
    "sort_by": "kd_value"        # Sort top binders by KD value
  },
  "performance": {
    "timeout_per_sequence": 10,   # Timeout per sequence (seconds)
    "max_retries": 2,            # Max retries for failed predictions
    "parallel_processing": false  # Enable parallel processing
  }
}
```

## Configuration Hierarchy

Settings are applied in the following order (later settings override earlier ones):

1. **Built-in defaults** (in script code)
2. **default_config.json** (global settings)
3. **Script-specific config** (e.g., predict_permeability_config.json)
4. **CLI arguments** (--limit, --helm_column, etc.)
5. **kwargs in function calls** (programmatic overrides)

## Customization Examples

### Processing Larger Datasets
```json
{
  "processing": {
    "limit": 1000,              # Process 1000 sequences
    "timeout_seconds": 600      # 10 minute timeout
  },
  "performance": {
    "parallel_processing": true, # Enable parallel processing
    "batch_size": 100           # Process in batches of 100
  }
}
```

### Custom Thresholds
```json
{
  "prediction": {
    "binding_thresholds": {
      "strong": 0.5,            # More stringent strong binding
      "moderate": 5.0,          # More stringent moderate binding
      "weak": 50.0              # More stringent weak binding
    }
  }
}
```

### Development Settings
```json
{
  "global": {
    "verbosity": "debug",       # Debug level logging
    "error_handling": "strict"  # Fail fast on errors
  },
  "processing": {
    "limit": 10,                # Small test datasets
    "validate_input": true      # Extra validation
  },
  "output": {
    "include_metadata": true,   # Include debugging metadata
    "include_timestamps": true  # Include timing information
  }
}
```

### Production Settings
```json
{
  "global": {
    "verbosity": "warn",        # Minimal logging
    "error_handling": "graceful" # Continue on errors
  },
  "processing": {
    "limit": 10000,             # Large production datasets
    "timeout_seconds": 1800     # 30 minute timeout
  },
  "performance": {
    "parallel_processing": true, # Use all CPU cores
    "cache_loaded_models": true # Cache for efficiency
  }
}
```

## Validation

Configuration files are validated when loaded:

- **Required fields** must be present
- **Numeric ranges** are checked (e.g., limit > 0)
- **File paths** are validated if specified
- **Enum values** are checked (e.g., format must be "csv")

## Environment Variables

Some settings can be overridden with environment variables:

```bash
export CYCPEP_MCP_LIMIT=50              # Default sequence limit
export CYCPEP_MCP_TIMEOUT=300           # Default timeout
export CYCPEP_MCP_MODEL_PATH=/path/to/models # Model directory
export CYCPEP_MCP_VERBOSE=true          # Enable verbose logging
```

## Troubleshooting

### Invalid JSON
```bash
# Validate JSON syntax
python -m json.tool configs/config_file.json
```

### Missing Required Fields
Check that required fields are present:
- `input.format`
- `processing.limit`
- `output.format`

### File Path Issues
- Use relative paths from repository root
- Ensure model files exist at specified paths
- Check file permissions for output directories

### Performance Issues
- Increase `timeout_seconds` for large datasets
- Enable `parallel_processing` for multi-core systems
- Adjust `batch_size` based on available memory