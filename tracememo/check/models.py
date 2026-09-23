"""Finding and report models for the grounding checker."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["error", "warning"]
CheckName = Literal["raw_number", "unknown_reference", "stale", "comparison", "claim"]


class Finding(BaseModel):
    """One problem found by a check."""

    check: CheckName
    severity: Severity
    file: str
    line: int
    text: str
    message: str
    ids: list[str] = Field(default_factory=list)

    def format(self) -> str:
        """One-line human-readable form: ``file:line: [check] message``."""
        return f"{self.file}:{self.line}: {self.severity} [{self.check}] {self.message}"


class CheckReport(BaseModel):
    """Result of running the checker over one or more files."""

    created: datetime
    files: list[str]
    findings: list[Finding] = Field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        """Findings with severity ``error``."""
        return [f for f in self.findings if f.severity == "error"]

    @property
    def passed(self) -> bool:
        """True when no check produced an error."""
        return not self.errors

    def counts(self) -> dict[str, int]:
        """Number of findings per check name."""
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.check] = out.get(f.check, 0) + 1
        return out

    def summary(self) -> str:
        """Human-readable summary block."""
        lines = [f.format() for f in self.findings]
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.counts().items())) or "none"
        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"check {status}: {len(self.errors)} error(s), findings: {counts}")
        return "\n".join(lines)
