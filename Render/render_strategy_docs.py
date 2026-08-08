"""Render strategy.md for every passing strategy in results/.

Reads each results/strategy_<N>/strategy.json (params + stage metrics) and
writes a human-readable, MQL5-portable strategy.md next to it. Useful for
re-rendering archived runs without re-running the GA.

Run from the repo root:
    python Render/render_strategy_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root even though this is in a subdir.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from composable.composable import ComposableStrategy
from Render.strategy_md import render_strategy_md

RESULTS_DIR = ROOT / "results"


def render_existing(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Render strategy.md for every strategy_<N>/strategy.json found."""
    rendered: list[Path] = []
    for folder in sorted(results_dir.glob("strategy_*")):
        json_path = folder / "strategy.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        params = data["params"]
        strat = ComposableStrategy(**params)
        metrics_map = {
            "htf_train": data.get("htf_train", {}),
            "m1_train": data.get("m1_train", {}),
            "m1_oos1": data.get("m1_oos1", {}),
            "m1_oos2": data.get("m1_oos2", {}),
        }
        md = render_strategy_md(folder, params, strat, metrics_map)
        rendered.append(md)
        print(f"rendered: {md}")
    return rendered


if __name__ == "__main__":
    render_existing()