"""MCP Server for Cyclic Peptide Tools

Provides both synchronous and asynchronous (submit) APIs for all tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
CONFIGS_DIR = MCP_ROOT / "configs"
# Python 3.7 environment for HELM-GPT repo scripts (optimization, etc.)
PY37_PATH = str(MCP_ROOT / "env_py3.7" / "bin" / "python")
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from utils import load_config, format_error_response, format_success_response
from loguru import logger

# Create MCP server
mcp = FastMCP("cycpep-tools")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted cyclic peptide computation job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed cyclic peptide computation job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running cyclic peptide computation job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted cyclic peptide computation jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

@mcp.tool()
def cleanup_completed_jobs(keep_days: int = 7) -> dict:
    """
    Clean up completed jobs older than specified days.

    Args:
        keep_days: Number of days to keep completed jobs (default: 7)

    Returns:
        Dictionary with cleanup results
    """
    return job_manager.cleanup_completed_jobs(keep_days)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def helm_to_smiles(
    helm_input: str = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    helm_column: str = "HELM",
    limit: Optional[int] = None
) -> dict:
    """
    Convert cyclic peptide sequences from HELM notation to SMILES representation.

    Fast operation - returns results immediately (typically ~33 sequences/second).

    Args:
        helm_input: HELM notation string to convert (used if input_file not provided)
        input_file: Optional CSV file path with HELM column to process in batch
        output_file: Optional path to save results CSV
        helm_column: Name of HELM column in CSV (default: "HELM")
        limit: Maximum number of sequences to process (default: from config)

    Returns:
        Dictionary with conversion results including:
        - status: "success" or "error"
        - results: List of conversion results for each input
        - success_count: Number of successful conversions
        - total_count: Total number of sequences processed
        - output_file: Path to saved CSV (if requested)
        - metadata: Execution metadata
    """
    try:
        # Load config
        config = load_config(str(CONFIGS_DIR / "helm_to_smiles_config.json"))

        # Import script function
        from helm_to_smiles import run_helm_to_smiles

        # Prepare arguments
        kwargs = {
            "helm_column": helm_column,
        }

        if limit is not None:
            kwargs["limit"] = limit
        elif "limit" in config:
            kwargs["limit"] = config["limit"]

        # Call the script function
        if input_file:
            result = run_helm_to_smiles(
                input_file=input_file,
                output_file=output_file,
                config=config,
                **kwargs
            )
        else:
            if not helm_input:
                return format_error_response("Either helm_input or input_file must be provided")

            # Process single HELM string (pass as input_file - it will be treated as single string)
            result = run_helm_to_smiles(
                input_file=helm_input,
                output_file=output_file,
                config=config,
                **kwargs
            )

        return format_success_response(result)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return format_error_response(f"File not found: {e}")
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return format_error_response(f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"HELM to SMILES conversion failed: {e}")
        return format_error_response(f"HELM to SMILES conversion failed: {str(e)}")

@mcp.tool()
def predict_permeability(
    helm_input: str = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    helm_column: str = "HELM",
    limit: Optional[int] = None,
    include_statistics: bool = True
) -> dict:
    """
    Predict cell membrane permeability for cyclic peptides using Random Forest model.

    Fast operation - returns predictions immediately (typically ~33 sequences/second).

    Args:
        helm_input: HELM notation string to predict (used if input_file not provided)
        input_file: Optional CSV file path with HELM column to process in batch
        output_file: Optional path to save results CSV
        helm_column: Name of HELM column in CSV (default: "HELM")
        limit: Maximum number of sequences to process (default: from config)
        include_statistics: Include summary statistics (default: True)

    Returns:
        Dictionary with permeability predictions including:
        - status: "success" or "error"
        - predictions: List of {helm, smiles, permeability_score, interpretation} for each input
        - summary: Statistics about permeability distribution
        - model_info: Information about the prediction model
        - output_file: Path to saved CSV (if requested)
    """
    try:
        # Load config
        config = load_config(str(CONFIGS_DIR / "predict_permeability_config.json"))

        # Import script function
        from predict_permeability import run_predict_permeability

        # Prepare arguments
        kwargs = {
            "helm_column": helm_column,
            "include_statistics": include_statistics
        }

        if limit is not None:
            kwargs["limit"] = limit
        elif "limit" in config:
            kwargs["limit"] = config["limit"]

        # Call the script function
        if input_file:
            result = run_predict_permeability(
                input_file=input_file,
                output_file=output_file,
                config=config,
                **kwargs
            )
        else:
            if not helm_input:
                return format_error_response("Either helm_input or input_file must be provided")

            # Process single HELM string (pass as input_file - it will be treated as single string)
            result = run_predict_permeability(
                input_file=helm_input,
                output_file=output_file,
                config=config,
                **kwargs
            )

        return format_success_response(result)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return format_error_response(f"File not found: {e}")
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return format_error_response(f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"Permeability prediction failed: {e}")
        return format_error_response(f"Permeability prediction failed: {str(e)}")

@mcp.tool()
def predict_kras_binding(
    helm_input: str = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    helm_column: str = "HELM",
    limit: Optional[int] = None,
    include_statistics: bool = True,
    top_binders_count: int = 5
) -> dict:
    """
    Predict KRAS protein binding affinity (KD values in μM) for cyclic peptides.

    Fast operation - returns predictions immediately (typically ~33 sequences/second).

    Args:
        helm_input: HELM notation string to predict (used if input_file not provided)
        input_file: Optional CSV file path with HELM column to process in batch
        output_file: Optional path to save results CSV
        helm_column: Name of HELM column in CSV (default: "HELM")
        limit: Maximum number of sequences to process (default: from config)
        include_statistics: Include summary statistics (default: True)
        top_binders_count: Number of top binders to highlight (default: 5)

    Returns:
        Dictionary with KRAS binding predictions including:
        - status: "success" or "error"
        - predictions: List of {helm, smiles, binding_score, kd_value_um, interpretation} for each input
        - summary: Statistics about binding distribution and top binders
        - model_info: Information about the prediction model
        - output_file: Path to saved CSV (if requested)
    """
    try:
        # Load config
        config = load_config(str(CONFIGS_DIR / "predict_kras_binding_config.json"))

        # Import script function
        from predict_kras_binding import run_predict_kras_binding

        # Prepare arguments
        kwargs = {
            "helm_column": helm_column,
            "include_statistics": include_statistics,
            "top_binders_count": top_binders_count
        }

        if limit is not None:
            kwargs["limit"] = limit
        elif "limit" in config:
            kwargs["limit"] = config["limit"]

        # Call the script function
        if input_file:
            result = run_predict_kras_binding(
                input_file=input_file,
                output_file=output_file,
                config=config,
                **kwargs
            )
        else:
            if not helm_input:
                return format_error_response("Either helm_input or input_file must be provided")

            # Process single HELM string (pass as input_file - it will be treated as single string)
            result = run_predict_kras_binding(
                input_file=helm_input,
                output_file=output_file,
                config=config,
                **kwargs
            )

        return format_success_response(result)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return format_error_response(f"File not found: {e}")
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return format_error_response(f"Invalid input: {e}")
    except Exception as e:
        logger.error(f"KRAS binding prediction failed: {e}")
        return format_error_response(f"KRAS binding prediction failed: {str(e)}")

# ==============================================================================
# Optimization Tools (RL-based peptide optimization)
# ==============================================================================

@mcp.tool()
def submit_optimize_peptides(
    prior_model: str,
    task: str,
    output_dir: str,
    agent_model: Optional[str] = None,
    n_steps: int = 500,
    batch_size: int = 32,
    sigma: float = 60,
    learning_rate: float = 1e-4,
    loss_type: str = "reinvent_cpl",
    alpha: float = 1.0,
    max_len: int = 140,
    device: str = "auto",
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a peptide optimization job using reinforcement learning.

    This tool uses HELM-GPT's agent training framework to optimize cyclic peptides
    for specific properties like permeability or KRAS binding affinity.

    IMPORTANT: This is a long-running operation. Use job management tools to track progress.

    Args:
        prior_model: Path to pre-trained prior model (.pt file) - REQUIRED
        task: Optimization task - REQUIRED. Options:
            - "permeability": Optimize for cell membrane permeability
            - "kras_kd": Optimize for KRAS protein binding affinity
            - "kras_perm": Multi-objective optimization for both properties
        output_dir: Directory to save results and model checkpoints - REQUIRED
        agent_model: Optional path to existing agent model for continued training
        n_steps: Number of optimization steps (default: 500)
        batch_size: Batch size for sequence generation (default: 32)
        sigma: Augmented likelihood sigma value (default: 60)
        learning_rate: Learning rate for optimization (default: 1e-4)
        loss_type: Loss function type - "reinvent", "cpl", or "reinvent_cpl" (default)
        alpha: Alpha parameter for CPL loss when using reinvent_cpl (default: 1.0)
        max_len: Maximum sequence length (default: 140)
        device: Device for training - "auto", "cpu", or "cuda" (default: auto)
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see training logs and scores

    Example:
        >>> result = submit_optimize_peptides(
        ...     prior_model="models/prior.pt",
        ...     task="permeability",
        ...     output_dir="./results/perm_opt",
        ...     n_steps=1000
        ... )
        >>> print(f"Job submitted: {result['job_id']}")
    """
    # Validate task
    valid_tasks = ["permeability", "kras_kd", "kras_perm"]
    if task not in valid_tasks:
        return format_error_response(f"Invalid task '{task}'. Must be one of: {valid_tasks}")

    # Validate prior model exists
    prior_path = Path(prior_model)
    if not prior_path.exists():
        return format_error_response(f"Prior model not found: {prior_model}")

    # Validate agent model if provided
    if agent_model:
        agent_path = Path(agent_model)
        if not agent_path.exists():
            return format_error_response(f"Agent model not found: {agent_model}")

    script_path = str(SCRIPTS_DIR / "optimize_peptides.py")

    args = {
        "prior": prior_model,
        "task": task,
        "output_dir": output_dir,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "loss_type": loss_type,
        "alpha": alpha,
        "max_len": max_len,
        "device": device
    }

    if agent_model:
        args["agent"] = agent_model

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"optimize_{task}",
        python_path=PY37_PATH
    )


@mcp.tool()
def check_optimization_requirements() -> dict:
    """
    Check if all requirements for peptide optimization are met.

    This verifies:
    - HELM-GPT repository availability
    - Scoring model availability (permeability, KRAS)
    - CUDA/GPU availability
    - Python environment

    Returns:
        Dictionary with availability status for each requirement:
        - repo_available: Whether HELM-GPT repo is accessible
        - permeability_model: Whether permeability scoring model exists
        - kras_model: Whether KRAS scoring model exists
        - cuda_available: Whether CUDA/GPU is available
        - python_version: Current Python version
        - available_prior_models: List of found prior model files
    """
    try:
        from optimize_peptides import check_optimization_requirements as check_reqs
        requirements = check_reqs()
        return format_success_response(requirements)
    except Exception as e:
        logger.error(f"Failed to check optimization requirements: {e}")
        return format_error_response(f"Failed to check requirements: {str(e)}")


@mcp.tool()
def get_optimization_tasks() -> dict:
    """
    Get information about available optimization tasks and their configurations.

    Returns:
        Dictionary with details about each optimization task:
        - permeability: Cell membrane permeability optimization
        - kras_kd: KRAS binding affinity optimization
        - kras_perm: Multi-objective optimization (both properties)
    """
    return format_success_response({
        "tasks": {
            "permeability": {
                "description": "Optimize cyclic peptides for cell membrane permeability",
                "scoring_functions": ["permeability"],
                "weights": [1.0],
                "model": "Random Forest (regression_rf.pkl)",
                "interpretation": "Higher scores indicate better permeability"
            },
            "kras_kd": {
                "description": "Optimize cyclic peptides for KRAS protein binding affinity",
                "scoring_functions": ["kras_kd"],
                "weights": [1.0],
                "model": "XGBoost (kras_xgboost_reg.pkl)",
                "interpretation": "Lower KD values indicate stronger binding"
            },
            "kras_perm": {
                "description": "Multi-objective: Optimize for both KRAS binding and permeability",
                "scoring_functions": ["permeability", "kras_kd"],
                "weights": [1.0, 1.0],
                "score_type": "sum",
                "interpretation": "Balanced optimization for both properties"
            }
        },
        "default_parameters": {
            "n_steps": 500,
            "batch_size": 32,
            "sigma": 60,
            "learning_rate": 1e-4,
            "loss_type": "reinvent_cpl",
            "alpha": 1.0,
            "max_len": 140
        },
        "loss_types": {
            "reinvent": "Standard REINVENT loss for molecular optimization",
            "cpl": "Comparative Preference Learning loss",
            "reinvent_cpl": "Combined REINVENT + CPL loss (recommended)"
        }
    })


# ==============================================================================
# Submit Tools (for long-running batch operations)
# ==============================================================================

@mcp.tool()
def submit_helm_to_smiles_batch(
    input_file: str,
    output_file: Optional[str] = None,
    helm_column: str = "HELM",
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a large HELM to SMILES conversion job for background processing.

    Use this for very large files that may take more than 10 minutes to process.
    For typical workloads (<1000 sequences), use the synchronous helm_to_smiles tool instead.

    Args:
        input_file: CSV file path with HELM column to process
        output_file: Optional path to save results CSV
        helm_column: Name of HELM column in CSV (default: "HELM")
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "helm_to_smiles.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input_file": input_file,
            "output": output_file,
            "helm_column": helm_column
        },
        job_name=job_name
    )

@mcp.tool()
def submit_permeability_batch(
    input_file: str,
    output_file: Optional[str] = None,
    helm_column: str = "HELM",
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a large permeability prediction job for background processing.

    Use this for very large files that may take more than 10 minutes to process.
    For typical workloads (<1000 sequences), use the synchronous predict_permeability tool instead.

    Args:
        input_file: CSV file path with HELM column to process
        output_file: Optional path to save results CSV
        helm_column: Name of HELM column in CSV (default: "HELM")
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "predict_permeability.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input_file": input_file,
            "output": output_file,
            "helm_column": helm_column
        },
        job_name=job_name
    )

@mcp.tool()
def submit_kras_binding_batch(
    input_file: str,
    output_file: Optional[str] = None,
    helm_column: str = "HELM",
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a large KRAS binding prediction job for background processing.

    Use this for very large files that may take more than 10 minutes to process.
    For typical workloads (<1000 sequences), use the synchronous predict_kras_binding tool instead.

    Args:
        input_file: CSV file path with HELM column to process
        output_file: Optional path to save results CSV
        helm_column: Name of HELM column in CSV (default: "HELM")
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "predict_kras_binding.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input_file": input_file,
            "output": output_file,
            "helm_column": helm_column
        },
        job_name=job_name
    )

# ==============================================================================
# Server-Based Scoring Tools (Boltz2/Rosetta)
# ==============================================================================

@mcp.tool()
def score_with_server(
    helm_input: str = None,
    input_file: Optional[str] = None,
    scorer_type: str = "rosetta",
    server_host: Optional[str] = None,
    server_api: Optional[str] = None,
    target_scores: Optional[str] = None,
    output_file: Optional[str] = None,
    timeout: Optional[int] = None,
    include_raw_scores: bool = True,
    limit: Optional[int] = None
) -> dict:
    """
    Score peptides using external Boltz2 or Rosetta server.

    This tool provides access to server-based computational scoring including:
    - Boltz2: Structure prediction with iPTM, iPAE, and affinity (pIC50) scoring
    - Rosetta: Energy-based scoring for ddG (binding), SAP (aggregation), CMS (contacts)

    IMPORTANT: Requires a running scoring server. Use get_server_scorer_info() to check
    configuration and check_server_connection() functionality.

    Args:
        helm_input: Single HELM or amino acid sequence to score (if input_file not provided)
        input_file: CSV file path with sequences to score in batch
        scorer_type: 'boltz2' or 'rosetta' (default: rosetta)
        server_host: Server URL (e.g., 'http://localhost:8001'). If not provided,
                    uses default from config.
        server_api: API endpoint. Defaults to:
                   - Boltz2: /biology/mit/boltz2/predict
                   - Rosetta: /rosetta/score
        target_scores: JSON string of score configurations. Example:
                      '{"ddG": {"weight": 1.0}, "SAP": {"weight": 0.5}}'
        output_file: Optional path to save results CSV
        timeout: Request timeout in seconds (default: 300 for Boltz2, 600 for Rosetta)
        include_raw_scores: Include raw scores before transformation (default: True)
        limit: Maximum number of sequences to process

    Returns:
        Dictionary with scoring results:
        - status: 'success', 'partial', or 'error'
        - results: Scoring results (single dict or list)
        - summary: Statistics about scoring
        - scorer_info: Configuration used
        - output_file: Path to saved CSV (if requested)

    Example:
        >>> result = score_with_server(
        ...     helm_input="PEPTIDE1{A.C.D.E.F.G.H.I.K}$$$$",
        ...     scorer_type="rosetta",
        ...     server_host="http://localhost:8001"
        ... )
        >>> print(f"ddG score: {result['results']['raw_ddG']}")
    """
    try:
        # Load config
        config = load_config(str(CONFIGS_DIR / "server_scoring_config.json"))

        # Import script function
        from score_with_server import run_server_scoring

        # Validate scorer type
        valid_scorers = ['boltz2', 'rosetta']
        if scorer_type not in valid_scorers:
            return format_error_response(f"Invalid scorer_type '{scorer_type}'. Must be one of: {valid_scorers}")

        # Parse target_scores JSON if provided
        parsed_target_scores = None
        if target_scores:
            try:
                parsed_target_scores = json.loads(target_scores)
            except json.JSONDecodeError as e:
                return format_error_response(f"Invalid target_scores JSON: {e}")

        # Build server config from loaded config
        scorer_config = config.get(scorer_type, {})

        # Prepare arguments
        kwargs = {
            "scorer_type": scorer_type,
            "include_raw_scores": include_raw_scores,
        }

        if server_host:
            kwargs["server_host"] = server_host
        elif "server" in scorer_config:
            kwargs["server_host"] = scorer_config["server"].get("host")

        if server_api:
            kwargs["server_api"] = server_api

        if timeout:
            kwargs["timeout"] = timeout
        elif "server" in scorer_config:
            kwargs["timeout"] = scorer_config["server"].get("timeout")

        if parsed_target_scores:
            kwargs["target_scores"] = parsed_target_scores
        elif "default_target_scores" in scorer_config:
            kwargs["target_scores"] = scorer_config["default_target_scores"]

        if limit is not None:
            kwargs["limit"] = limit

        # Build server_config for the scorer
        kwargs["server_config"] = scorer_config

        # Call the script function
        if input_file:
            result = run_server_scoring(
                input_file=input_file,
                output_file=output_file,
                **kwargs
            )
        else:
            if not helm_input:
                return format_error_response("Either helm_input or input_file must be provided")

            # Process single sequence
            result = run_server_scoring(
                input_file=helm_input,
                output_file=output_file,
                **kwargs
            )

        return format_success_response(result)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return format_error_response(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Server scoring failed: {e}")
        return format_error_response(f"Server scoring failed: {str(e)}")


@mcp.tool()
def submit_server_scoring(
    input_file: str,
    scorer_type: str,
    output_dir: str,
    server_host: Optional[str] = None,
    server_config_file: Optional[str] = None,
    target_scores: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit server scoring job for background processing (large batches).

    Use this for large batch scoring jobs that may take more than 10 minutes.
    For smaller batches (<100 sequences), use score_with_server instead.

    Args:
        input_file: CSV file path with sequences to score
        scorer_type: 'boltz2' or 'rosetta'
        output_dir: Directory to save results
        server_host: Server URL (optional, uses config default)
        server_config_file: Path to YAML config file (optional)
        target_scores: JSON string of score configurations
        job_name: Optional name for tracking

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see logs
    """
    # Validate scorer type
    valid_scorers = ['boltz2', 'rosetta']
    if scorer_type not in valid_scorers:
        return format_error_response(f"Invalid scorer_type '{scorer_type}'. Must be one of: {valid_scorers}")

    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        return format_error_response(f"Input file not found: {input_file}")

    # Build arguments
    script_path = str(SCRIPTS_DIR / "score_with_server.py")

    args = {
        "input": input_file,
        "scorer": scorer_type,
        "output": str(Path(output_dir) / f"{scorer_type}_scores.csv")
    }

    if server_host:
        args["server"] = server_host

    if server_config_file:
        args["config"] = server_config_file

    if target_scores:
        args["target-scores"] = target_scores

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"server_scoring_{scorer_type}",
        python_path=PY37_PATH
    )


@mcp.tool()
def get_server_scorer_info() -> dict:
    """
    Get information about available server scorers and their configuration.

    Returns details about:
    - Available scorer types (Boltz2, Rosetta)
    - Supported scores for each scorer
    - Default configurations
    - Score transformation options
    - Server connection requirements

    Use this to understand what scoring options are available before
    calling score_with_server or submit_server_scoring.

    Returns:
        Dictionary with:
        - available_scorers: List of scorer types
        - boltz2: Boltz2 scorer configuration and supported scores
        - rosetta: Rosetta scorer configuration and supported scores
        - transformation_types: Available score transformation functions
        - usage_examples: Example calls for each scorer
    """
    try:
        # Load server scoring config
        config = load_config(str(CONFIGS_DIR / "server_scoring_config.json"))

        # Try to get repo availability info
        try:
            sys.path.insert(0, str(SCRIPTS_DIR))
            from lib.server_scoring_utils import get_available_server_scorers
            repo_info = get_available_server_scorers()
        except Exception:
            repo_info = {"repo_available": False}

        return format_success_response({
            "available_scorers": ["boltz2", "rosetta"],
            "repo_implementation_available": repo_info.get("repo_available", False),
            "boltz2": {
                "description": "Structure prediction and affinity scoring using Boltz2",
                "default_server": config.get("boltz2", {}).get("server", {}),
                "supported_scores": config.get("boltz2", {}).get("supported_scores", []),
                "score_descriptions": config.get("boltz2", {}).get("score_descriptions", {}),
                "default_target_scores": config.get("boltz2", {}).get("default_target_scores", {}),
                "typical_timeout": "300 seconds"
            },
            "rosetta": {
                "description": "Energy-based scoring for binding and aggregation using Rosetta",
                "default_server": config.get("rosetta", {}).get("server", {}),
                "supported_scores": config.get("rosetta", {}).get("supported_scores", []),
                "score_descriptions": config.get("rosetta", {}).get("score_descriptions", {}),
                "default_target_scores": config.get("rosetta", {}).get("default_target_scores", {}),
                "typical_timeout": "600 seconds"
            },
            "transformation_types": config.get("transformation_types", {}),
            "usage_examples": {
                "rosetta_single": {
                    "tool": "score_with_server",
                    "args": {
                        "helm_input": "PEPTIDE1{A.C.D.E.F.G.H.I.K}$$$$",
                        "scorer_type": "rosetta",
                        "server_host": "http://localhost:8001"
                    }
                },
                "boltz2_batch": {
                    "tool": "submit_server_scoring",
                    "args": {
                        "input_file": "sequences.csv",
                        "scorer_type": "boltz2",
                        "output_dir": "./results"
                    }
                },
                "custom_target_scores": {
                    "tool": "score_with_server",
                    "args": {
                        "input_file": "sequences.csv",
                        "scorer_type": "rosetta",
                        "target_scores": '{"ddG": {"weight": 1.0}, "SAP": {"weight": 0.5}}'
                    }
                }
            },
            "notes": [
                "Server-based scoring requires external Boltz2/Rosetta servers running",
                "Use the mock server (scripts/mock_scoring_server.py) for testing",
                "Rosetta scoring is typically slower than Boltz2 (600s vs 300s timeout)",
                "For production use, configure target_scores based on your optimization goals"
            ]
        })

    except Exception as e:
        logger.error(f"Failed to get server scorer info: {e}")
        return format_error_response(f"Failed to get server scorer info: {str(e)}")


# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_helm_notation(helm_input: str) -> dict:
    """
    Validate HELM notation format for cyclic peptides.

    Args:
        helm_input: HELM notation string to validate

    Returns:
        Dictionary with validation result:
        - valid: Boolean indicating if HELM is valid
        - error: Error message if invalid
        - structure_info: Information about the peptide structure
    """
    try:
        from helm_to_smiles import run_helm_to_smiles

        # Try to convert to validate (pass as input_file - it will be treated as single string)
        result = run_helm_to_smiles(input_file=helm_input)

        if result.get("results") and len(result["results"]) > 0:
            conversion = result["results"][0]
            return format_success_response({
                "valid": conversion.get("success", False),
                "helm": conversion.get("helm_sequence"),
                "smiles": conversion.get("smiles"),
                "error": conversion.get("error") if not conversion.get("success") else None
            })
        else:
            return format_error_response("No conversion result received")

    except Exception as e:
        logger.error(f"HELM validation failed: {e}")
        return format_error_response(f"HELM validation failed: {str(e)}")

@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the MCP server and available tools.

    Returns:
        Dictionary with server information, tool list, and system status
    """
    return {
        "status": "success",
        "server_name": "cycpep-tools",
        "version": "1.2.0",
        "description": "MCP server for cyclic peptide computational tools including RL-based optimization and server-based scoring (Boltz2/Rosetta)",
        "sync_tools": [
            "helm_to_smiles",
            "predict_permeability",
            "predict_kras_binding",
            "validate_helm_notation",
            "score_with_server"
        ],
        "server_scoring_tools": [
            "score_with_server",
            "submit_server_scoring",
            "get_server_scorer_info"
        ],
        "optimization_tools": [
            "submit_optimize_peptides",
            "check_optimization_requirements",
            "get_optimization_tasks"
        ],
        "submit_tools": [
            "submit_helm_to_smiles_batch",
            "submit_permeability_batch",
            "submit_kras_binding_batch",
            "submit_server_scoring"
        ],
        "job_management": [
            "get_job_status",
            "get_job_result",
            "get_job_log",
            "cancel_job",
            "list_jobs",
            "cleanup_completed_jobs"
        ],
        "config_directory": str(CONFIGS_DIR),
        "scripts_directory": str(SCRIPTS_DIR),
        "models_directory": str(MCP_ROOT / "examples" / "data" / "models")
    }

@mcp.tool()
def get_model_info() -> dict:
    """
    Get information about the loaded prediction and optimization models.

    Returns:
        Dictionary with model details and availability for:
        - Scoring models (permeability, KRAS binding)
        - Prior models for optimization (if available)
    """
    models_dir = MCP_ROOT / "examples" / "data" / "models"
    repo_dir = MCP_ROOT / "repo" / "helm-gpt"

    model_info = {
        "status": "success",
        "models_directory": str(models_dir),
        "scoring_models": {},
        "prior_models": []
    }

    # Check permeability scoring model
    perm_model_path = models_dir / "regression_rf.pkl"
    model_info["scoring_models"]["permeability"] = {
        "path": str(perm_model_path),
        "available": perm_model_path.exists(),
        "type": "Random Forest Regressor",
        "purpose": "Cell membrane permeability prediction",
        "size_mb": round(perm_model_path.stat().st_size / (1024*1024), 2) if perm_model_path.exists() else 0
    }

    # Check KRAS binding scoring model
    kras_model_path = models_dir / "kras_xgboost_reg.pkl"
    model_info["scoring_models"]["kras_binding"] = {
        "path": str(kras_model_path),
        "available": kras_model_path.exists(),
        "type": "XGBoost Regressor",
        "purpose": "KRAS protein binding affinity prediction",
        "size_mb": round(kras_model_path.stat().st_size / (1024*1024), 2) if kras_model_path.exists() else 0
    }

    # Check for prior models (needed for optimization)
    if repo_dir.exists():
        prior_models = list(repo_dir.glob("**/*.pt"))
        for pm in prior_models[:10]:  # Limit to first 10
            try:
                model_info["prior_models"].append({
                    "path": str(pm),
                    "name": pm.name,
                    "size_mb": round(pm.stat().st_size / (1024*1024), 2)
                })
            except Exception:
                pass

    model_info["optimization_note"] = (
        "To run optimization (submit_optimize_peptides), you need a prior model (.pt file). "
        "Prior models can be trained using the HELM-GPT train_prior.py script or downloaded "
        "from the HELM-GPT repository releases."
    )

    return model_info

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    logger.info("Starting CycPep Tools MCP Server")
    mcp.run()