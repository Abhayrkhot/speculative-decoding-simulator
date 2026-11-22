"""Regenerate every figure in docs/ (and the numbers quoted in the README).

    python scripts/make_figures.py [results/*.json]

Simulator-only figures are always produced. If benchmark results are given, the
calibration and validation figures are produced from the first file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from specsim import HARDWARE, MODELS, ConstantLatency, RooflineLatency, expected_speedup, simulate
from specsim.calibrate import calibration_report, validate
from specsim.plots import (
    acceptance_plot,
    batch_scaling_plot,
    sim_vs_theory_plot,
    speedup_heatmap,
    validation_plot,
)

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main(argv: list[str]) -> None:
    DOCS.mkdir(exist_ok=True)
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

    if argv:
        res_path = argv[0]
        cal = calibration_report(res_path)
        acceptance_plot(cal, str(DOCS / "acceptance_by_position.png"))
        rep = validate(res_path)
        validation_plot(rep, str(DOCS / "validation.png"))
        (DOCS / "validation.json").write_text(json.dumps(rep, indent=2))
        (DOCS / "calibration.json").write_text(json.dumps(cal, indent=2))
    print(f"figures written to {DOCS}")


if __name__ == "__main__":
    main(sys.argv[1:])
