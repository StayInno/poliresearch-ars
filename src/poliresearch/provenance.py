"""Provenance logging — the MLflow + DVC role from the deck (slides 12-13), TRIZ #25 Self-service.

Every run records its question, corpus hash, model, and per-claim verdicts to a JSONL file so
results are reproducible and auditable. If MLflow is installed it is also logged there; if not,
the JSONL log is fully sufficient. No silent failures — everything is on disk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, runs_dir: str | Path, run_id: str | None = None):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or time.strftime("run-%Y%m%d-%H%M%S")
        self.path = self.runs_dir / f"{self.run_id}.jsonl"
        self._mlflow = self._try_mlflow()

    @staticmethod
    def _try_mlflow():
        try:
            import mlflow  # type: ignore
            return mlflow
        except Exception:
            return None

    def event(self, kind: str, **payload: Any) -> None:
        record = {"ts": time.time(), "run_id": self.run_id, "kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self._mlflow is not None and kind == "metric":
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    try:
                        self._mlflow.log_metric(k, v)
                    except Exception:
                        pass

    def params(self, **params: Any) -> None:
        self.event("params", **params)
        if self._mlflow is not None:
            try:
                self._mlflow.log_params({k: str(v) for k, v in params.items()})
            except Exception:
                pass
