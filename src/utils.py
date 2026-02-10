"""Shared utilities for MCP server."""

from pathlib import Path
from typing import Dict, Any, Optional
import json
from loguru import logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}")
        return {}

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

def validate_input_file(file_path: str) -> bool:
    """Validate that input file exists and is readable."""
    if not file_path:
        return False
    path = Path(file_path)
    return path.exists() and path.is_file()

def format_error_response(error: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Format standardized error response."""
    response = {
        "status": "error",
        "error": str(error)
    }
    if context:
        response["context"] = context
    return response

def format_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format standardized success response."""
    return {
        "status": "success",
        **data
    }