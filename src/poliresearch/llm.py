"""LLM provider interface with two backends.

The rest of the system depends only on `LLM.complete(...)`, so swapping providers is one method.

  * ClaudeCodeLLM  — shells out to the local `claude` CLI in print mode (`claude -p`). Uses your
                     existing Claude Code authentication, so NO API key is needed. This is the
                     default whenever the `claude` CLI is on PATH and no API key is set.
  * AnthropicLLM   — calls the Anthropic API directly via the `anthropic` SDK (needs a key).

`make_llm` auto-selects: an explicit POLIRESEARCH_LLM_BACKEND wins; otherwise an API key picks
the SDK, else the `claude` CLI is used if present.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Protocol


class LLM(Protocol):
    available: bool
    model: str
    def complete(self, system: str, prompt: str, *, max_tokens: int = 1500) -> str: ...


def _to_cli_model(model: str) -> str:
    """Map a full model id to a Claude Code CLI alias (the CLI prefers aliases)."""
    m = model.lower()
    if "opus" in m:
        return "opus"
    if "sonnet" in m:
        return "sonnet"
    if "haiku" in m:
        return "haiku"
    return model


class ClaudeCodeLLM:
    """Run completions through the authenticated Claude Code CLI — no API key required."""

    def __init__(self, model: str = "claude-opus-4-8", binary: str = "claude",
                 timeout_s: int = 75, max_retries: int = 2, backoff_base: float = 1.5):
        self.model = model
        self._cli_model = _to_cli_model(model)
        self._binary = shutil.which(binary)
        self.available = self._binary is not None
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1500) -> str:
        if not self.available:
            raise RuntimeError(
                "Claude Code CLI ('claude') not found on PATH. Install Claude Code, or set "
                "ANTHROPIC_API_KEY to use the API backend."
            )
        # No CLI flag for system prompts in print mode, so prepend it to the user turn.
        full = f"{system}\n\n{prompt}" if system else prompt
        cmd = [self._binary, "-p", "--output-format", "json", "--model", self._cli_model]
        last_err = ""
        # The CLI occasionally returns a transient exit 1 (rate limit / flake). Retry with
        # backoff so a single transient failure does not abort a long evaluation.
        for attempt in range(self.max_retries + 1):
            try:
                # Force UTF-8 I/O: real corpora contain Unicode (em/en dashes, smart quotes),
                # and the Windows locale codec (cp1252) cannot encode them on the subprocess pipe.
                proc = subprocess.run(cmd, input=full, capture_output=True, text=True,
                                      encoding="utf-8", errors="replace", timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                last_err = f"timeout after {self.timeout_s}s"
            else:
                if proc.returncode == 0:
                    try:
                        data = json.loads(proc.stdout)
                    except json.JSONDecodeError:
                        return proc.stdout.strip()  # fall back to raw text
                    if not data.get("is_error"):
                        return str(data.get("result", "")).strip()
                    last_err = f"is_error: {data.get('result') or data}"
                else:
                    last_err = (proc.stderr or proc.stdout or "no output").strip()[:300]
            if attempt < self.max_retries:
                time.sleep(self.backoff_base * (2 ** attempt))
        raise RuntimeError(f"claude CLI failed after {self.max_retries + 1} attempts: {last_err}")


class AnthropicLLM:
    def __init__(self, api_key: str | None, model: str = "claude-opus-4-8"):
        self.model = model
        self._client = None
        self.available = False
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
                self.available = True
            except Exception:
                self.available = False

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1500) -> str:
        if not self.available or self._client is None:
            raise RuntimeError("No Anthropic API key configured.")
        msg = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def resolve_backend(settings) -> str:
    """Decide which backend to use: explicit setting > API key > local claude CLI."""
    explicit = getattr(settings, "llm_backend", None)
    if explicit:
        return explicit
    if settings.anthropic_api_key:
        return "anthropic"
    if shutil.which("claude"):
        return "claude_code"
    return "anthropic"  # nothing available; AnthropicLLM will report unavailable


def make_llm(settings, model: str | None = None) -> LLM:
    """Build an LLM client. Pass `model` to override settings.model (e.g. to give the
    Falsifier a different model from the Generator)."""
    chosen = model or settings.model
    backend = resolve_backend(settings)
    if backend == "claude_code":
        return ClaudeCodeLLM(chosen)
    return AnthropicLLM(settings.anthropic_api_key, chosen)
