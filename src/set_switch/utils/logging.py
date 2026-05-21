"""Logging setup and local metric logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)


class JsonlMetricLogger:
    """Append training/eval metrics to a local JSONL file."""

    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **metrics: Any) -> None:
        if not self.enabled:
            return
        row = {"time": time.time(), **metrics}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
