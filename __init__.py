"""S3 Integration for ComfyUI - Enhanced Version

A comprehensive S3 integration package with unified nodes, robust error handling,
and improved logging capabilities.
"""

import os
import sys
from typing import Dict, Any

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import logging configuration first
from .logging_config import get_logger, set_log_level

# Initialize logger
logger = get_logger("S3Integration")

try:
    # Import node mappings
    from .node_mappings import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    # Import individual nodes for direct access
    from .s3_config_node import S3ConfigNode
    from .s3_image_node import S3ImageNode
    from .s3_video_node import S3VideoNode
    
    # Import client and utilities
    from .s3_client import S3Client, get_s3_client, S3ClientError
    
    logger.info("S3 Integration nodes loaded successfully")
    
except ImportError as e:
    logger.error(f"Failed to import S3 Integration components: {e}")
    # Fallback to empty mappings to prevent ComfyUI from crashing
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

except Exception as e:
    logger.error(f"Unexpected error during S3 Integration initialization: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}





# Package metadata
__version__ = "2.0.0"
__author__ = "S3 Integration Team"
__description__ = "Enhanced S3 Integration for ComfyUI with unified nodes and robust error handling"

# Configuration helpers
def configure_logging(level: str = "INFO", enable_file_logging: bool = False, log_file: str = None) -> None:
    """Configure logging for S3 Integration
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to enable file logging
        log_file: Path to log file (optional)
    """
    set_log_level(level)
    
    if enable_file_logging:
        os.environ["S3_LOG_TO_FILE"] = "true"
        if log_file:
            os.environ["S3_LOG_FILE_PATH"] = log_file
    
    logger.info(f"Logging configured: level={level}, file_logging={enable_file_logging}")


def get_package_info() -> Dict[str, Any]:
    """Get package information
    
    Returns:
        Dictionary with package metadata
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "nodes": list(NODE_CLASS_MAPPINGS.keys()),

    }


# Environment validation
def validate_environment() -> Dict[str, Any]:
    """Validate S3 environment configuration
    
    Returns:
        Dictionary with validation results
    """
    required_vars = ["S3_REGION", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET_NAME"]
    optional_vars = ["S3_ENDPOINT_URL", "S3_INPUT_DIR", "S3_OUTPUT_DIR"]
    
    results = {
        "valid": True,
        "missing_required": [],
        "missing_optional": [],
        "configured_vars": {}
    }
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            results["missing_required"].append(var)
            results["valid"] = False
        else:
            results["configured_vars"][var] = "***" if "KEY" in var else value
    
    # Check optional variables
    for var in optional_vars:
        value = os.getenv(var)
        if not value:
            results["missing_optional"].append(var)
        else:
            results["configured_vars"][var] = value
    
    return results


# Auto-validate environment on import
if os.getenv("S3_AUTO_VALIDATE", "true").lower() == "true":
    validation = validate_environment()
    if not validation["valid"]:
        logger.warning(f"S3 environment validation failed. Missing: {validation['missing_required']}")
    else:
        logger.debug("S3 environment validation passed")


# Export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "S3ConfigNode",
    "S3ImageNode",
    "S3VideoNode",
    "S3Client",
    "get_s3_client",
    "S3ClientError",
    "configure_logging",
    "get_package_info",
    "validate_environment",
]

logger.info(f"S3 Integration v{__version__} initialized with {len(NODE_CLASS_MAPPINGS)} nodes")