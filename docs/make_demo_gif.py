"""Render docs/images/demo.gif: a terminal-style animation of the drone example build.

Runs `tracememo synth drone` and `tracememo build` on the drone example, captures their
output, and draws it line by line with Pillow (no screen recording needed).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images" / "demo.gif"
COMMANDS = [
    ["tracememo", "synth", "drone", "--out", "data/synthetic/drone", "--seed", "0"],
    ["tracememo", "build", "--config", "examples/drone_memo/project.yaml"],
    [
        "tracememo",
        "explain",
        "drone.loc_err.mean_cm",
        "--config",
        "examples/drone_memo/project.yaml",
    ],
]
W, H, PAD, LINE_H, FONT_SIZE, MAX_LINES = 880, 520, 16, 18, 13, 26
BG, FG, DIM, PROMPT, OK = (
    (22, 22, 24),
    (232, 232, 226),
    (137, 135, 129),
    (57, 135, 229),
    (25, 158, 112),
)


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("Menlo.ttc", "DejaVuSansMono.ttf", "Consolas.ttf", "SFMono-Regular.otf"):
        for base in (
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path("/usr/share/fonts/truetype/dejavu"),
            Path.home() / "Library/Fonts",
        ):
            f = base / name
            if f.exists():
                return ImageFont.truetype(str(f), FONT_SIZE)
    return ImageFont.load_default()


def capture() -> list[tuple[str, tuple[int, int, int]]]:
    """Run the demo commands and return (line, color) pairs, shortening long paths."""
    lines: list[tuple[str, tuple[int, int, int]]] = []
    for cmd in COMMANDS:
        lines.append(("$ " + " ".join(cmd), PROMPT))
        proc = subprocess.run(
            [str(ROOT / ".venv" / "bin" / cmd[0]), *cmd[1:]],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        for ln in proc.stdout.splitlines():
            ln = ln.replace(str(ROOT) + "/", "")
            color = OK if ("PASSED" in ln or ln.startswith("wrote")) else FG
            lines.append((ln[:118], color))
        lines.append(("", FG))
    return lines


def render(lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    font = _font()
    frames: list[Image.Image] = []
    durations: list[int] = []
    # One frame per prompt line, otherwise one frame per two lines, to keep the file small.
    steps = [
        n
        for n in range(1, len(lines) + 1)
        if n == len(lines) or lines[n - 1][0].startswith("$") or n % 2 == 0
    ]
    for n in steps:
        visible = lines[max(0, n - MAX_LINES) : n]
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, W, 28), radius=0, fill=(40, 40, 44))
        for k, c in enumerate(((255, 95, 87), (255, 189, 46), (39, 201, 63))):
            d.ellipse((12 + 20 * k, 8, 24 + 20 * k, 20), fill=c)
        d.text((W // 2 - 60, 6), "tracememo demo", fill=DIM, font=font)
        for i, (text, color) in enumerate(visible):
            d.text((PAD, 40 + i * LINE_H), text, fill=color, font=font)
        frames.append(im)
        durations.append(900 if text.startswith("$") else 160)
    durations[-1] = 4000
    palette = [f.quantize(colors=32, method=Image.Quantize.MEDIANCUT) for f in frames]
    palette[0].save(
        OUT, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True
    )


if __name__ == "__main__":
    render(capture())
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    sys.exit(0)
