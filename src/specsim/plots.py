"""Static figures for reports (requires ``matplotlib``: ``pip install specsim[plot]``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .analytic import expected_speedup, optimal_gamma
from .latency import LatencyModel

# Validated categorical order (fixed, never cycled) and a single-hue sequential ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TASK_ORDER = ["code", "math", "chat", "summarize"]


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "lines.linewidth": 2,
        }
    )
    return plt


def _task_color(task: str) -> str:
    i = TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)
    return SERIES[i % len(SERIES)]


def _save(fig: Any, path: str) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def speedup_heatmap(
    alphas: Sequence[float],
    gammas: Sequence[int],
    latency: LatencyModel,
    path: str,
    batch: int = 1,
    title: str | None = None,
) -> None:
    """Expected speedup over alpha x gamma, with the best gamma per alpha outlined."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    grid = np.array([[expected_speedup(a, g, latency, batch) for g in gammas] for a in alphas])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(gammas), 0.9 + 0.32 * len(alphas)))
    ax.grid(False)
    im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(gammas)), [str(g) for g in gammas])
    ax.set_yticks(range(len(alphas)), [f"{a:.2f}" for a in alphas])
    ax.set_xlabel("draft tokens per cycle (gamma)")
    ax.set_ylabel("acceptance rate (alpha)")
    for i, row in enumerate(grid):
        j = int(np.argmax(row))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
        light = row[j] > grid.min() + 0.6 * (grid.max() - grid.min())
        ax.text(
            j,
            i,
            f"{row[j]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color=SURFACE if light else INK,
        )
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("expected speedup (x)", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title(title or "Expected speedup (outlined: best gamma for each alpha)", loc="left")
    _save(fig, path)


def validation_plot(report: dict[str, Any], path: str) -> None:
    """Measured vs predicted speedup by gamma, one panel per temperature."""
    plt = _plt()
    rows = report["rows"]
    temps = sorted({r["temperature"] for r in rows})
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in rows)]
    tasks += sorted({r["task"] for r in rows} - set(tasks))
    fig, axes = plt.subplots(
        1, len(temps), figsize=(5.2 * len(temps), 3.6), sharey=True, squeeze=False
    )
    for ax, temp in zip(axes[0], temps, strict=True):
        ax.axhline(1.0, color=INK_2, lw=0.8, ls=(0, (2, 2)))
        for task in tasks:
            rs = sorted(
                (r for r in rows if r["task"] == task and r["temperature"] == temp),
                key=lambda r: r["gamma"],
            )
            if not rs:
                continue
            g = [r["gamma"] for r in rs]
            col = _task_color(task)
            ax.plot(
                g,
                [r["measured"] for r in rs],
                color=col,
                marker="o",
                ms=5,
                mec=SURFACE,
                mew=1.5,
                label=f"{task} measured",
            )
            ax.plot(
                g,
                [r["predicted"] for r in rs],
                color=col,
                lw=1.2,
                ls=(0, (4, 2)),
                label=f"{task} predicted",
            )
            ax.annotate(
                task,
                (g[-1], rs[-1]["measured"]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK_2,
            )
        label = "greedy" if temp == 0 else f"T = {temp:g}"
        ax.set_title(f"{label}: measured (solid) vs predicted (dashed)", loc="left")
        ax.set_xlabel("draft tokens per cycle (gamma)")
    axes[0][0].set_ylabel("decode speedup vs target-only (x)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, -0.12)
    )
    s = report["summary"]
    fig.suptitle(
        f"{report['meta']['draft']} -> {report['meta']['target']} on {report['meta']['device']}"
        f"   (mean |error| {s['mae_pct']:.1f}%)",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK_2,
    )
    _save(fig, path)


def acceptance_plot(calibration: dict[str, Any], path: str) -> None:
    """Conditional acceptance probability by draft position, one panel per temperature."""
    plt = _plt()
    acc = calibration["acceptance"]
    temps = sorted({a["temperature"] for a in acc})
    fig, axes = plt.subplots(
        1, len(temps), figsize=(4.6 * len(temps), 3.3), sharey=True, squeeze=False
    )
    for ax, temp in zip(axes[0], temps, strict=True):
        for a in sorted(
            (a for a in acc if a["temperature"] == temp),
            key=lambda a: TASK_ORDER.index(a["task"]) if a["task"] in TASK_ORDER else 99,
        ):
            pos = np.arange(1, len(a["positional"]) + 1)
            ax.plot(
                pos,
                a["positional"],
                color=_task_color(a["task"]),
                marker="o",
                ms=4,
                mec=SURFACE,
                mew=1.2,
                label=f"{a['task']} (mean {a['alpha']:.2f})",
            )
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("draft position")
        label = "greedy" if temp == 0 else f"T = {temp:g}"
        ax.set_title(f"Acceptance by draft position, {label}", loc="left")
        ax.legend(loc="lower left")
    axes[0][0].set_ylabel("P(accept | earlier drafts accepted)")
    _save(fig, path)


def batch_scaling_plot(
    latency_for_alpha: LatencyModel,
    alphas: Sequence[float],
    batches: Sequence[int],
    path: str,
    title: str,
    max_gamma: int = 16,
) -> None:
    """Best achievable speedup vs batch size under a (roofline) latency model."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(1.0, color=INK_2, lw=0.8, ls=(0, (2, 2)))
    for i, a in enumerate(alphas):
        best = [optimal_gamma(a, latency_for_alpha, max_gamma, b)[1] for b in batches]
        col = ORDINAL_BLUE[min(i, len(ORDINAL_BLUE) - 1)]
        ax.plot(
            batches,
            best,
            color=col,
            marker="o",
            ms=4,
            mec=SURFACE,
            mew=1.2,
            label=f"alpha = {a:.2f}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(batches), [str(b) for b in batches])
    ax.set_xlabel("batch size (sequences decoded together)")
    ax.set_ylabel("best speedup over gamma (x)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right")
    _save(fig, path)


def sim_vs_theory_plot(
    rows: Sequence[dict[str, float]], path: str, title: str = "Monte Carlo vs closed form"
) -> None:
    """``rows`` items: alpha, analytic, simulated, ci_low, ci_high (one gamma)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    a = [r["alpha"] for r in rows]
    ax.plot(a, [r["analytic"] for r in rows], color=SERIES[0], label="closed form")
    ax.errorbar(
        a,
        [r["simulated"] for r in rows],
        yerr=[
            [r["simulated"] - r["ci_low"] for r in rows],
            [r["ci_high"] - r["simulated"] for r in rows],
        ],
        fmt="o",
        ms=5,
        color=SERIES[1],
        mec=SURFACE,
        mew=1.2,
        capsize=0,
        label="Monte Carlo (95% CI)",
    )
    ax.set_xlabel("acceptance rate (alpha)")
    ax.set_ylabel("speedup (x)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    _save(fig, path)


def combined_validation_plot(
    reports: Sequence[tuple[str, dict[str, Any]]], path: str, title: str | None = None
) -> None:
    """Predicted vs measured speedup for every configuration of several runs."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    allv = [v for _, rep in reports for r in rep["rows"] for v in (r["measured"], r["predicted"])]
    lo, hi = min(allv) * 0.95, max(allv) * 1.05
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.axvline(1.0, color=GRID, lw=1.0, zorder=0)
    ax.axhline(1.0, color=GRID, lw=1.0, zorder=0)
    for i, (label, rep) in enumerate(reports):
        m = [r["measured"] for r in rep["rows"]]
        p = [r["predicted"] for r in rep["rows"]]
        s = rep["summary"]
        ax.scatter(
            m,
            p,
            s=26,
            color=SERIES[i % len(SERIES)],
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
            label=f"{label} (mean |err| {s['mae_pct']:.1f}%)",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("measured decode speedup (x)")
    ax.set_ylabel("predicted speedup (x)")
    ax.set_title(title or "Predicted vs measured, every task x temperature x gamma", loc="left")
    ax.legend(loc="upper left")
    ax.text(
        hi - 0.03 * (hi - lo), hi - 0.08 * (hi - lo), "y = x", ha="right", fontsize=8, color=INK_2
    )
    _save(fig, path)
