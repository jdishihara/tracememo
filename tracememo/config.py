"""Project configuration (``project.yaml``) schema and loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    """One raw input source handled by an adapter."""

    adapter: str
    path: str
    tables: list[str] | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    """An analysis to run, with parameter overrides."""

    id: str
    params: dict[str, Any] = Field(default_factory=dict)


class ReportConfig(BaseModel):
    """Report template and output settings."""

    markdown_template: str = "report.md.j2"
    latex_template: str = "report.tex"
    formats: list[Literal["markdown", "latex"]] = Field(default_factory=lambda: ["markdown"])
    output_name: str = "report"
    provenance_links: bool = False
    compile_pdf: bool = True  # only if latexmk is installed


class CheckConfig(BaseModel):
    """Grounding checker settings."""

    run_in_build: bool = True
    files: list[str] = Field(default_factory=list)  # extra templates/fragments to check
    allow_patterns: list[str] = Field(default_factory=list)  # extra regexes for raw numbers
    llm_claims: bool = False  # also run the LLM-assisted claim check (costs API calls)


class LLMConfig(BaseModel):
    """LLM settings. The model name is never hard-coded elsewhere."""

    model: str = "claude-sonnet-5"
    max_tokens: int = 4096


class DraftConfig(BaseModel):
    """Where drafted section fragments are written."""

    out_dir: str = "drafts"
    syntax: Literal["markdown", "latex", "placeholder"] = "markdown"
    max_fix_rounds: int = 1


class ProjectConfig(BaseModel):
    """Top-level project configuration."""

    name: str
    build_dir: str = "build"
    inputs: list[InputConfig] = Field(default_factory=list)
    analyses: list[AnalysisConfig] = Field(default_factory=list)
    report: ReportConfig = Field(default_factory=ReportConfig)
    check: CheckConfig = Field(default_factory=CheckConfig)
    draft: DraftConfig = Field(default_factory=DraftConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    config_dir: Path = Field(default=Path("."), exclude=True)

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path from the config relative to the config file's directory."""
        p = Path(path)
        return p if p.is_absolute() else (self.config_dir / p).resolve()

    @property
    def build_path(self) -> Path:
        """Absolute build directory."""
        return self.resolve(self.build_dir)

    @property
    def manifest_path(self) -> Path:
        """Path of ``build/manifest.json``."""
        return self.build_path / "manifest.json"


def load_config(path: Path) -> ProjectConfig:
    """Load and validate a ``project.yaml`` file."""
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ProjectConfig.model_validate(raw)
    cfg.config_dir = path.parent
    return cfg
