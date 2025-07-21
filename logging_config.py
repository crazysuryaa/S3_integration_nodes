"""Enhanced logging configuration for S3 Integration

Provides structured logging with configurable levels, formatters, and outputs.
"""

import os
import sys
import copy
import logging
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    COLORS = {
        "DEBUG": "\033[0;36m",      # CYAN
        "INFO": "\033[0;32m",       # GREEN
        "WARNING": "\033[0;33m",    # YELLOW
        "ERROR": "\033[0;31m",      # RED
        "CRITICAL": "\033[0;37;41m", # WHITE ON RED
        "RESET": "\033[0m",         # RESET COLOR
    }
    
    def format(self, record):
        """Format log record with colors"""
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for machine-readable logs"""
    
    def format(self, record):
        """Format log record with structured information"""
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        # Add module information
        if hasattr(record, 'module'):
            record.component = record.module
        else:
            record.component = record.name
        
        return super().format(record)


class LoggerConfig:
    """Logger configuration manager"""
    
    DEFAULT_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    DETAILED_FORMAT = "[%(timestamp)s] [%(component)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s"
    SIMPLE_FORMAT = "[%(name)s] - %(levelname)s - %(message)s"
    
    def __init__(self):
        self.log_level = self._get_log_level()
        self.log_format = self._get_log_format()
        self.enable_file_logging = self._get_file_logging_enabled()
        self.log_file_path = self._get_log_file_path()
    
    def _get_log_level(self) -> int:
        """Get log level from environment or default"""
        level_str = os.getenv("S3_LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_str, logging.INFO)
    
    def _get_log_format(self) -> str:
        """Get log format from environment or default"""
        format_type = os.getenv("S3_LOG_FORMAT", "simple").lower()
        
        if format_type == "detailed":
            return self.DETAILED_FORMAT
        elif format_type == "default":
            return self.DEFAULT_FORMAT
        else:
            return self.SIMPLE_FORMAT
    
    def _get_file_logging_enabled(self) -> bool:
        """Check if file logging is enabled"""
        return os.getenv("S3_LOG_TO_FILE", "false").lower() == "true"
    
    def _get_log_file_path(self) -> str:
        """Get log file path"""
        default_path = os.path.join("logs", "s3_integration.log")
        return os.getenv("S3_LOG_FILE_PATH", default_path)


def setup_logger(name: str, config: Optional[LoggerConfig] = None) -> logging.Logger:
    """Set up a logger with the given configuration
    
    Args:
        name: Logger name
        config: Logger configuration (uses default if None)
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = LoggerConfig()
    
    logger = logging.getLogger(name)
    logger.setLevel(config.log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.propagate = False
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(config.log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if enabled
    if config.enable_file_logging:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(config.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(config.log_file_path)
            file_formatter = StructuredFormatter(config.DETAILED_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")
    
    return logger


# Global logger instances
_loggers = {}
_config = LoggerConfig()


def get_logger(name: str = "S3Integration") -> logging.Logger:
    """Get or create a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = setup_logger(name, _config)
    
    return _loggers[name]


# Default logger for backward compatibility
logger = get_logger("S3Integration")


def set_log_level(level: str) -> None:
    """Set log level for all loggers
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    for logger_instance in _loggers.values():
        logger_instance.setLevel(log_level)


def enable_debug_logging() -> None:
    """Enable debug logging for all loggers"""
    set_log_level("DEBUG")


def disable_debug_logging() -> None:
    """Disable debug logging (set to INFO level)"""
    set_log_level("INFO")