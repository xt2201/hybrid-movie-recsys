"""
Centralized logging module with Rich formatting and file output.
Provides professional logging with progress bars, tables, and metrics display.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

# Custom theme for the project
CUSTOM_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "metric": "bold magenta",
    "step": "bold blue",
})

# Global console instance
console = Console(theme=CUSTOM_THEME)

# Logger instances cache
_loggers: Dict[str, logging.Logger] = {}

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


def setup_file_handler(logger: logging.Logger, log_file: str) -> None:
    """Setup file handler for a logger without size limit."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # File handler - no rotation, unlimited size
    file_handler = logging.FileHandler(
        LOG_DIR / log_file,
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Detailed format for file
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def get_logger(
    name: str = "hybrid_recsys",
    log_file: Optional[str] = None,
    level: int = logging.DEBUG
) -> logging.Logger:
    """
    Get or create a logger with Rich console output and file logging.
    
    Args:
        name: Logger name (usually module name)
        log_file: Optional specific log file name. Defaults to 'hybrid_recsys.log'
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    
    # File handler
    log_filename = log_file or "hybrid_recsys.log"
    setup_file_handler(logger, log_filename)
    
    _loggers[name] = logger
    return logger


class TrainingProgress:
    """
    Professional training progress manager with Rich progress bars.
    Displays metrics, progress, and training statistics.
    """
    
    def __init__(
        self,
        total_iterations: int,
        description: str = "Training",
        logger: Optional[logging.Logger] = None
    ):
        self.total = total_iterations
        self.description = description
        self.logger = logger or get_logger()
        self.metrics: Dict[str, float] = {}
        self.start_time = None
        
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10,
        )
        self.task_id = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        self.logger.info(f"🚀 Started {self.description} ({self.total} iterations)")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
        elapsed = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"✅ {self.description} completed in {elapsed}")
            if self.metrics:
                self._log_final_metrics()
        else:
            self.logger.error(f"❌ {self.description} failed after {elapsed}")
        
        return False
    
    def update(self, advance: int = 1, **metrics):
        """Update progress and optionally log metrics."""
        self.progress.update(self.task_id, advance=advance)
        
        if metrics:
            self.metrics.update(metrics)
            # Update task description with latest metric
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.progress.update(
                self.task_id,
                description=f"{self.description} [{metric_str}]"
            )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to file and optionally display."""
        self.metrics.update(metrics)
        step_str = f"Step {step}: " if step is not None else ""
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.debug(f"{step_str}{metric_str}")
    
    def _log_final_metrics(self):
        """Display final metrics in a beautiful table."""
        table = Table(title="📊 Final Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        for name, value in self.metrics.items():
            table.add_row(name, f"{value:.4f}")
        
        console.print(table)


class EvaluationProgress:
    """
    Progress tracker for model evaluation with detailed metrics.
    """
    
    def __init__(
        self,
        total_users: int,
        description: str = "Evaluating",
        logger: Optional[logging.Logger] = None
    ):
        self.total = total_users
        self.description = description
        self.logger = logger or get_logger()
        
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=10,
        )
        self.task_id = None
    
    def __enter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
        return False
    
    def update(self, advance: int = 1):
        """Update progress."""
        self.progress.update(self.task_id, advance=advance)


def log_header(title: str, logger: Optional[logging.Logger] = None):
    """Display a prominent header for major sections."""
    logger = logger or get_logger()
    console.print()
    console.print(Panel(f"[bold white]{title}[/bold white]", style="blue", expand=False))
    logger.info(f"{'='*60}")
    logger.info(f"  {title}")
    logger.info(f"{'='*60}")


def log_metrics_table(
    metrics: Dict[str, float],
    title: str = "Metrics",
    logger: Optional[logging.Logger] = None
):
    """Display metrics in a formatted table."""
    logger = logger or get_logger()
    
    table = Table(title=f"📊 {title}", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", justify="right", width=15)
    
    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.4f}")
        else:
            table.add_row(name, str(value))
    
    console.print(table)
    
    # Also log to file
    logger.info(f"--- {title} ---")
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")


def log_config(config: Dict[str, Any], title: str = "Configuration", logger: Optional[logging.Logger] = None):
    """Display configuration in a formatted panel."""
    logger = logger or get_logger()
    
    from rich.pretty import Pretty
    console.print(Panel(Pretty(config), title=f"⚙️ {title}", expand=False))
    
    # Log to file
    logger.info(f"--- {title} ---")
    _log_dict_recursive(config, logger, indent=0)


def _log_dict_recursive(d: Dict, logger: logging.Logger, indent: int = 0):
    """Recursively log dictionary to file."""
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            logger.info(f"{prefix}{key}:")
            _log_dict_recursive(value, logger, indent + 1)
        else:
            logger.info(f"{prefix}{key}: {value}")


def log_model_summary(
    model_name: str,
    params: Dict[str, Any],
    logger: Optional[logging.Logger] = None
):
    """Display model summary with parameters."""
    logger = logger or get_logger()
    
    table = Table(title=f"🤖 Model: {model_name}", show_header=True, header_style="bold blue")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow", justify="right")
    
    for name, value in params.items():
        table.add_row(name, str(value))
    
    console.print(table)
    
    logger.info(f"Model: {model_name}")
    for name, value in params.items():
        logger.info(f"  {name}: {value}")


def log_data_summary(
    n_users: int,
    n_items: int,
    n_interactions: int,
    sparsity: float,
    logger: Optional[logging.Logger] = None
):
    """Display dataset summary."""
    logger = logger or get_logger()
    
    table = Table(title="📁 Dataset Summary", show_header=True, header_style="bold green")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="yellow", justify="right")
    
    table.add_row("Users", f"{n_users:,}")
    table.add_row("Items", f"{n_items:,}")
    table.add_row("Interactions", f"{n_interactions:,}")
    table.add_row("Sparsity", f"{sparsity:.4%}")
    
    console.print(table)
    
    logger.info("Dataset Summary:")
    logger.info(f"  Users: {n_users:,}")
    logger.info(f"  Items: {n_items:,}")
    logger.info(f"  Interactions: {n_interactions:,}")
    logger.info(f"  Sparsity: {sparsity:.4%}")


def log_success(message: str, logger: Optional[logging.Logger] = None):
    """Log success message with checkmark."""
    logger = logger or get_logger()
    console.print(f"[success]✅ {message}[/success]")
    logger.info(f"SUCCESS: {message}")


def log_warning(message: str, logger: Optional[logging.Logger] = None):
    """Log warning message."""
    logger = logger or get_logger()
    console.print(f"[warning]⚠️ {message}[/warning]")
    logger.warning(message)


def log_error(message: str, logger: Optional[logging.Logger] = None):
    """Log error message."""
    logger = logger or get_logger()
    console.print(f"[error]❌ {message}[/error]")
    logger.error(message)


def log_step(step_num: int, total_steps: int, message: str, logger: Optional[logging.Logger] = None):
    """Log a numbered step in a process."""
    logger = logger or get_logger()
    console.print(f"[step]📌 Step {step_num}/{total_steps}: {message}[/step]")
    logger.info(f"Step {step_num}/{total_steps}: {message}")
