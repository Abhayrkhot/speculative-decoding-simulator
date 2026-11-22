"""Regenerate every figure in docs/ (and the numbers quoted in the README).

    python scripts/make_figures.py [results/*.json]

Simulator-only figures are always produced. Each benchmark result adds a
calibration/validation figure, and all of them together form the combined
predicted-vs-measured figure.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from specsim import HARDWARE, MODELS, ConstantLatency, RooflineLatency, expected_speedup, simulate
from specsim.calibrate import calibration_report, validate
from specsim.plots import (
    acceptance_plot,
    batch_scaling_plot,
    combined_validation_plot,
    sim_vs_theory_plot,
    speedup_heatmap,
    validation_plot,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"


def tag_of(path: str) -> str:
    """results/qwen2.5-0.5b_32b_graphs_12058393.json -> 0.5b_32b_graphs"""
    stem = Path(path).stem
    stem = re.sub(r"_\d+$", "", stem)
    return stem.split("-", 1)[1] if "-" in stem else stem


def label_of(tag: str) -> str:
    parts = tag.split("_")
    mode = "CUDA graphs" if "graphs" in parts else "eager"
    return f"{parts[0].upper()} -> {parts[1].upper()}, {mode}"


def simulator_figures() -> None:
    alphas = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    lat = ConstantLatency.from_ratio(0.05)
    speedup_heatmap(
        alphas,
        list(range(1, 11)),
        lat,
        str(DOCS / "heatmap_c005.png"),
        title="Expected speedup, draft cost c = 0.05 (outlined: best gamma)",
    )

    rows = []
    for a in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        r = simulate(a, 4, lat, tokens=256, trials=2000, seed=0)
        rows.append(
            {
                "alpha": a,
                "analytic": expected_speedup(a, 4, lat),
                "simulated": r.speedup,
                "ci_low": r.speedup_ci95[0],
                "ci_high": r.speedup_ci95[1],
            }
        )
    sim_vs_theory_plot(
        rows,
        str(DOCS / "sim_vs_theory.png"),
        title="Monte Carlo (256 tokens) vs closed form, gamma = 4, c = 0.05",
    )

    roof = RooflineLatency(
        MODELS["llama-3.1-70b"], MODELS["llama-3.2-1b"], HARDWARE["h100-sxm"], context=2048
    )
    batch_scaling_plot(
        roof,
        [0.6, 0.75, 0.9],
        [1, 2, 4, 8, 16, 32, 64, 128, 256],
        str(DOCS / "batch_scaling.png"),
        title="Roofline: Llama-3.1-70B + 1B draft on H100, best gamma",
    )


def main(argv: list[str]) -> None:
    DOCS.mkdir(exist_ok=True)
    simulator_figures()
    reports = []
    summary = {}
    for path in sorted(argv):
        tag = tag_of(path)
        cal = calibration_report(path)
        rep = validate(path)
        validation_plot(rep, str(DOCS / f"validation_{tag}.png"))
        if "32b" in tag and "graphs" not in tag:
            acceptance_plot(cal, str(DOCS / "acceptance_by_position.png"))
        reports.append((label_of(tag), rep))
        summary[tag] = {"calibration": cal, "validation": rep}
    if reports:
        order = {"eager": 0, "CUDA graphs": 1}
        reports.sort(key=lambda lr: (order[lr[0].split(", ")[1]], lr[0]))
        combined_validation_plot(reports, str(DOCS / "predicted_vs_measured.png"))
        (DOCS / "results_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"figures written to {DOCS}")


if __name__ == "__main__":
    main(sys.argv[1:])
