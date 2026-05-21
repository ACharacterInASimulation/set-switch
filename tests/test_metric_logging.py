from __future__ import annotations

import json

from set_switch.utils.logging import JsonlMetricLogger


def test_jsonl_metric_logger_writes_rows(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricLogger(path)

    logger.log(event="train", step=1, loss=0.5)
    logger.log(event="eval", step=2, val_loss=0.4)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "train"
    assert rows[0]["step"] == 1
    assert rows[0]["loss"] == 0.5
    assert rows[1]["event"] == "eval"
    assert rows[1]["val_loss"] == 0.4
    assert "time" in rows[0]
