"""Logging utilities."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", 
                 format_str: Optional[str] = None,
                 log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Configure handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_str))
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_str))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_str,
        handlers=handlers,
        force=True
    )
    
    # Suppress noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('scipy').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"performance.{name}")
        self.start_time = None
    
    def start(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Started: {operation}")
    
    def end(self, operation: str) -> float:
        """End timing an operation."""
        import time
        if self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        self.logger.info(f"Completed: {operation} in {elapsed:.3f}s")
        self.start_time = None
        return elapsed