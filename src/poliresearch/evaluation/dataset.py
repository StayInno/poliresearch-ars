"""Labeled evaluation datasets and loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..models import Reference


@dataclass
class CitationExample:
    id: str
    reference: Reference
    label: str            # "valid" (should be trustworthy) | "invalid" (should be rejected)
    note: str = ""

    @property
    def should_accept(self) -> bool:
        return self.label == "valid"


@dataclass
class ExperimentExample:
    id: str
    hypothesis: str
    code: str
    label: str            # "true" (code should exit 0) | "false"
    note: str = ""

    @property
    def should_accept(self) -> bool:
        return self.label == "true"


@dataclass
class FalsificationExample:
    id: str
    hypothesis: str
    label: str            # "survive" (corpus supports) | "refute" (corpus contradicts/silent)
    note: str = ""

    @property
    def should_accept(self) -> bool:
        return self.label == "survive"


def _read_json_list(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_citations(path: str | Path) -> list[CitationExample]:
    out = []
    for d in _read_json_list(path):
        ref = Reference(**d["reference"])
        out.append(CitationExample(id=d["id"], reference=ref, label=d["label"],
                                   note=d.get("note", "")))
    return out


def load_experiments(path: str | Path) -> list[ExperimentExample]:
    return [ExperimentExample(id=d["id"], hypothesis=d["hypothesis"], code=d["code"],
                              label=d["label"], note=d.get("note", ""))
            for d in _read_json_list(path)]


def load_falsification(path: str | Path) -> list[FalsificationExample]:
    return [FalsificationExample(id=d["id"], hypothesis=d["hypothesis"], label=d["label"],
                                 note=d.get("note", ""))
            for d in _read_json_list(path)]
