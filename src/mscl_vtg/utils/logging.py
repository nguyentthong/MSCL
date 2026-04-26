"""Simple logging helpers using rich."""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    table = Table(title=f"{prefix} Metrics" if prefix else "Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in sorted(metrics.items()):
        table.add_row(k, f"{v:.4f}")
    console.print(table)


def log_losses(losses: dict[str, float], epoch: int, step: int) -> None:
    parts = [f"[bold]E{epoch} S{step}[/bold]"]
    for k, v in losses.items():
        parts.append(f"{k}={v:.4f}")
    console.print("  ".join(parts))


def setup_file_logger(log_dir: str | Path) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    return log_path
