"""Experiment runner — the CS/AI superpower that moves the system past literature synthesis.

Biomedical systems (Robin, AI co-scientist) stop at *proposing* experiments a human must run in
a wet lab. CS/AI is computational, so the system can CLOSE THE LOOP: write code, run it, observe
the result, and let that empirical evidence confirm or refute a hypothesis. This is the
Coscientist `EXPERIMENT` action and Sakana's experiment loop, reduced to its safe core.

Empirical verification is strictly stronger than literature falsification: a hypothesis that is
*confirmed by a reproducible experiment* escapes the closed-corpus novelty ceiling (it can be
true even if no paper says so). That is the whole point of doing CS/AI first.

SECURITY: this executes code. The generated code is UNTRUSTED. Here we run it in a separate
process with a wall-clock timeout, a fresh temp working directory, and no arguments. That is NOT
a real sandbox — for anything beyond local trusted use, run inside a container / gVisor / seccomp
jail with no network and a read-only FS. The runner is deliberately small so that hardening is a
drop-in replacement of `_spawn`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentResult:
    ran: bool
    success: bool          # process exited 0 within the timeout
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False

    @property
    def verdict_line(self) -> str:
        if self.timed_out:
            return "experiment TIMED OUT"
        if not self.ran:
            return "experiment did not run"
        return f"experiment {'PASSED' if self.success else 'FAILED'} (exit {self.returncode})"


class ExperimentRunner:
    def __init__(self, timeout_s: int = 30, python: str | None = None):
        self.timeout_s = timeout_s
        self.python = python or sys.executable

    def run_python(self, code: str) -> ExperimentResult:
        """Run a self-contained Python snippet. Convention: the snippet should exit non-zero
        (e.g. via `assert` or sys.exit(1)) when the hypothesis it tests is FALSE."""
        with tempfile.TemporaryDirectory(prefix="poliresearch-exp-") as tmp:
            script = Path(tmp) / "experiment.py"
            script.write_text(code, encoding="utf-8")
            return self._spawn(script, cwd=tmp)

    def _spawn(self, script: Path, cwd: str) -> ExperimentResult:
        try:
            proc = subprocess.run(
                [self.python, str(script)],
                cwd=cwd, capture_output=True, text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return ExperimentResult(ran=True, success=False, stdout=e.stdout or "",
                                    stderr=e.stderr or "", returncode=None, timed_out=True)
        except Exception as e:  # spawn failure
            return ExperimentResult(ran=False, success=False, stdout="",
                                    stderr=f"failed to launch experiment: {e}", returncode=None)
        return ExperimentResult(
            ran=True, success=(proc.returncode == 0),
            stdout=proc.stdout[-4000:], stderr=proc.stderr[-4000:], returncode=proc.returncode,
        )
