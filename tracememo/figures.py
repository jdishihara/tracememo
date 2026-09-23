"""Matplotlib style and figure saving shared by all analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure as MplFigure  # noqa: E402

STYLE_PATH = Path(__file__).parent / "report" / "tracememo.mplstyle"

# Categorical series colors, in fixed order (never cycled past the list).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def use_style() -> None:
    """Activate the shared tracememo matplotlib style."""
    plt.style.use(str(STYLE_PATH))


def new_figure(width: float = 6.0, height: float = 3.2) -> tuple[MplFigure, plt.Axes]:
    """Create a styled figure and single axes."""
    use_style()
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def save_figure(fig: MplFigure, base_path: Path) -> tuple[Path, Path]:
    """Save a figure as ``<base>.pdf`` and ``<base>.png`` and close it."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    # Ids contain dots, so build the names by hand instead of using with_suffix().
    pdf = base_path.parent / (base_path.name + ".pdf")
    png = base_path.parent / (base_path.name + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png
