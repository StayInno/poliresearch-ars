"""Configuration. Loads from environment / a .env file with zero hard dependencies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we don't require python-dotenv. Silently ignores a
    missing file; existing environment variables always win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    llm_backend: str | None  # "claude_code" | "anthropic" | None (auto-detect)
    model: str            # generator model (OPEN phase)
    falsifier_model: str  # judge model (CLOSED phase) - deliberately different for independence
    falsifier_mode: str   # "debate" (default, best F1) | "decompose" | "vote"
    crossref_mailto: str | None
    corpus_dir: Path
    runs_dir: Path
    domain_key: str       # research domain profile; defaults to CS/AI

    @property
    def has_llm(self) -> bool:
        import shutil
        return bool(self.anthropic_api_key) or shutil.which("claude") is not None


def load_settings(env_file: str | os.PathLike | None = ".env") -> Settings:
    if env_file is not None:
        _load_dotenv(Path(env_file))
    return Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        llm_backend=os.environ.get("POLIRESEARCH_LLM_BACKEND") or None,
        model=os.environ.get("POLIRESEARCH_MODEL", "claude-opus-4-8"),
        # Default the judge to a different model than the generator so they don't share
        # correlated errors. Override with POLIRESEARCH_FALSIFIER_MODEL.
        falsifier_model=os.environ.get("POLIRESEARCH_FALSIFIER_MODEL", "claude-sonnet-4-6"),
        # Debate panel is the measured best (F1 0.86 vs 0.63 skeptic on the 47-example set).
        falsifier_mode=os.environ.get("POLIRESEARCH_FALSIFIER_MODE", "debate"),
        crossref_mailto=os.environ.get("CROSSREF_MAILTO") or None,
        corpus_dir=Path(os.environ.get("POLIRESEARCH_CORPUS", "./corpus")),
        runs_dir=Path(os.environ.get("POLIRESEARCH_RUNS", "./runs")),
        domain_key=os.environ.get("POLIRESEARCH_DOMAIN", "cs_ai"),
    )
