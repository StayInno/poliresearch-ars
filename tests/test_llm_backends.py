"""Offline tests for LLM backend selection and the Claude Code provider (subprocess mocked)."""

from __future__ import annotations

import subprocess
import types

from poliresearch import llm as llm_mod
from poliresearch.llm import ClaudeCodeLLM, _to_cli_model, make_llm, resolve_backend


def _settings(api_key=None, backend=None):
    return types.SimpleNamespace(anthropic_api_key=api_key, llm_backend=backend,
                                 model="claude-opus-4-8")


def test_model_alias_mapping():
    assert _to_cli_model("claude-opus-4-8") == "opus"
    assert _to_cli_model("claude-sonnet-4-6") == "sonnet"
    assert _to_cli_model("claude-haiku-4-5-20251001") == "haiku"
    assert _to_cli_model("some-custom-id") == "some-custom-id"


def test_resolve_backend_prefers_explicit():
    assert resolve_backend(_settings(api_key="k", backend="claude_code")) == "claude_code"


def test_resolve_backend_api_key_then_cli(monkeypatch):
    assert resolve_backend(_settings(api_key="k")) == "anthropic"
    monkeypatch.setattr(llm_mod.shutil, "which", lambda _: "/usr/bin/claude")
    assert resolve_backend(_settings(api_key=None)) == "claude_code"
    monkeypatch.setattr(llm_mod.shutil, "which", lambda _: None)
    assert resolve_backend(_settings(api_key=None)) == "anthropic"


def test_make_llm_builds_claude_code_when_no_key(monkeypatch):
    monkeypatch.setattr(llm_mod.shutil, "which", lambda _: "/usr/bin/claude")
    obj = make_llm(_settings(api_key=None), model="claude-sonnet-4-6")
    assert isinstance(obj, ClaudeCodeLLM)
    assert obj._cli_model == "sonnet"
    assert obj.available


def test_claude_code_complete_parses_json(monkeypatch):
    monkeypatch.setattr(llm_mod.shutil, "which", lambda _: "/usr/bin/claude")
    captured = {}

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None, **kw):
        captured["cmd"] = cmd
        captured["input"] = input
        return types.SimpleNamespace(
            returncode=0,
            stdout='{"is_error": false, "result": "PONG"}',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = ClaudeCodeLLM(model="claude-sonnet-4-6").complete("SYS", "USER")
    assert out == "PONG"
    assert "SYS" in captured["input"] and "USER" in captured["input"]  # system prepended
    assert "--model" in captured["cmd"] and "sonnet" in captured["cmd"]


def test_claude_code_complete_raises_on_error(monkeypatch):
    monkeypatch.setattr(llm_mod.shutil, "which", lambda _: "/usr/bin/claude")

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None, **kw):
        return types.SimpleNamespace(returncode=0,
                                     stdout='{"is_error": true, "result": "rate limited"}',
                                     stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    import pytest
    with pytest.raises(RuntimeError):
        ClaudeCodeLLM().complete("", "x")
