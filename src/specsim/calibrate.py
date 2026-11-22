"""Calibrate the simulator from real measurements and check its predictions.

Input is the JSON written by ``bench/spec_bench.py``: a forward-latency profile
for both models plus per-cycle acceptance traces and wall-clock times for
plain decoding (``gamma=0``) and speculative decoding at several ``gamma``.

``validate`` predicts each measured speedup *out of sample*:

* acceptance for a run (task, temperature, gamma, prompt) is fitted only on
  traces from *other prompts at other gammas* of the same task and
  temperature. Holding out the prompt matters at temperature 0, where every
  gamma decodes the same text and so sees the same draft/target agreement;
* latency is interpolated from the standalone forward-pass profile (between
  the profiled context lengths);
* host-side overhead per forward pass is estimated from the ``gamma=0`` runs.

Nothing is fitted to the speculative wall-clock times being predicted.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .acceptance import AcceptanceModel, IIDAcceptance, PositionalAcceptance
from .latency import LinearLatency, TabulatedLatency
from .simulate import simulate

__all__ = [
    "calibration_report",
    "fit_latency",
    "greedy_match",
    "load_results",
    "measured_speedups",
    "tabulated_latency",
    "validate",
]

Results = dict[str, Any]


def load_results(path: str | Path | Results) -> Results:
    if isinstance(path, dict):
        return path
    with open(path) as f:
        data: Results = json.load(f)
    for key in ("meta", "latency_ms", "runs"):
        if key not in data:
            raise ValueError(f"{path}: not a spec_bench result (missing {key!r})")
    return data


def _mean_context(res: Results) -> float:
    runs = [r for r in res["runs"] if r["gamma"] == 0] or res["runs"]
    return float(np.mean([r["prompt_tokens"] + r["n_new"] / 2 for r in runs]))


def fit_latency(res: Results, context: int | None = None) -> tuple[LinearLatency, int]:
    """Fit a :class:`LinearLatency` from the profile closest to ``context`` tokens."""
    profiles = {int(c): v for c, v in res["latency_ms"].items()}
    if not profiles:
        raise ValueError("results contain no latency profile")
    want = context if context is not None else _mean_context(res)
    ctx = min(profiles, key=lambda c: abs(c - want))
    prof = profiles[ctx]
    target = {int(k): ms / 1e3 for k, ms in prof["target"].items()}
    draft = {int(k): ms / 1e3 for k, ms in prof["draft"].items()}
    return LinearLatency.fit(target, draft), ctx


def tabulated_latency(res: Results, context: float | None = None) -> tuple[TabulatedLatency, int]:
    """Latency at ``context`` tokens, interpolated between the profiled contexts.

    Returns the model and the (rounded) context it represents. Outside the
    profiled range the nearest profile is used.
    """
    profiles = {int(c): v for c, v in res["latency_ms"].items()}
    if not profiles:
        raise ValueError("results contain no latency profile")
    want = float(context) if context is not None else _mean_context(res)
    ctxs = sorted(profiles)
    lo = max([c for c in ctxs if c <= want], default=ctxs[0])
    hi = min([c for c in ctxs if c >= want], default=ctxs[-1])
    w = 0.0 if hi == lo else (want - lo) / (hi - lo)

    def blend(model: str) -> dict[int, float]:
        a, b = profiles[lo][model], profiles[hi][model]
        return {int(k): ((1 - w) * a[k] + w * b[k]) / 1e3 for k in a if k in b}

    return TabulatedLatency.from_profiles(blend("target"), blend("draft")), round(
        lo + w * (hi - lo)
    )


def _groups(res: Results) -> dict[tuple[str, float], list[dict[str, Any]]]:
    g: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in res["runs"]:
        g[(r["task"], float(r["temperature"]))].append(r)
    return dict(g)


def _cycles(
    runs: Iterable[dict[str, Any]],
    exclude_gamma: int | None = None,
    exclude_prompt: int | None = None,
) -> list[list[int]]:
    out: list[list[int]] = []
    for r in runs:
        if r["gamma"] == 0 or r["gamma"] == exclude_gamma:
            continue
        if exclude_prompt is not None and r.get("prompt_id") == exclude_prompt:
            continue
        out.extend(r["cycles"])
    return out


def _per_token(runs: Iterable[dict[str, Any]]) -> float:
    runs = list(runs)
    n = sum(r["n_new"] for r in runs)
    return float(sum(r["decode_s"] for r in runs)) / float(n)


def measured_speedups(res: Results) -> dict[tuple[str, float, int], float]:
    """Decode throughput of each gamma relative to gamma=0, per (task, temperature)."""
    out: dict[tuple[str, float, int], float] = {}
    for (task, temp), runs in _groups(res).items():
        by_g: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in runs:
            by_g[int(r["gamma"])].append(r)
        if 0 not in by_g:
            continue
        base = _per_token(by_g[0])
        for g, rs in by_g.items():
            if g:
                out[(task, temp, g)] = base / _per_token(rs)
    return out


def calibration_report(path: str | Path | Results) -> dict[str, Any]:
    """Fitted latency parameters and acceptance statistics per task and temperature."""
    res = load_results(path)
    lat, ctx = fit_latency(res)
    acceptance = []
    for (task, temp), runs in sorted(_groups(res).items()):
        cyc = _cycles(runs)
        if not cyc:
            continue
        pos = PositionalAcceptance.from_traces(cyc)
        acceptance.append(
            {
                "task": task,
                "temperature": temp,
                "alpha": pos.mean_alpha,
                "positional": list(pos.values),
                "position_counts": list(pos.counts),
                "cycles": len(cyc),
            }
        )
    return {
        "meta": res["meta"],
        "latency": {
            "context": ctx,
            "target_fixed_ms": lat.target_fixed_s * 1e3,
            "target_per_token_ms": lat.target_per_token_s * 1e3,
            "draft_fixed_ms": lat.draft_fixed_s * 1e3,
            "draft_per_token_ms": lat.draft_per_token_s * 1e3,
            "c": lat.draft(1) / lat.target(1),
        },
        "acceptance": acceptance,
        "greedy_match": greedy_match(res),
    }


def greedy_match(res: Results) -> dict[str, float]:
    """Compare greedy speculative outputs with greedy target-only outputs.

    In exact arithmetic they are identical. In bf16, verifying ``gamma + 1``
    tokens in one pass can round differently from one-token decoding, so a
    near-tie can flip an argmax and the continuations diverge from there.
    """
    base = {
        (r["task"], r["prompt_id"]): r["tokens"]
        for r in res["runs"]
        if r["gamma"] == 0 and float(r["temperature"]) == 0.0
    }
    total = exact = 0
    prefix_frac: list[float] = []
    for r in res["runs"]:
        ref = base.get((r["task"], r["prompt_id"]))
        if r["gamma"] == 0 or float(r["temperature"]) != 0.0 or ref is None or not r["tokens"]:
            continue
        toks = r["tokens"]
        total += 1
        exact += toks == ref
        n = min(len(toks), len(ref))
        same = next((i for i in range(n) if toks[i] != ref[i]), n)
        prefix_frac.append(same / max(len(ref), 1))
    return {
        "runs": total,
        "exact_match_rate": exact / total if total else float("nan"),
        "mean_matching_prefix": float(np.mean(prefix_frac)) if prefix_frac else float("nan"),
    }


def validate(
    path: str | Path | Results,
    acceptance: str = "positional",
    trials: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Out-of-sample prediction of every measured speedup; see module docstring."""
    if acceptance not in ("positional", "iid"):
        raise ValueError("acceptance must be 'positional' or 'iid'")
    res = load_results(path)
    lat, ctx = tabulated_latency(res)
    measured = measured_speedups(res)
    groups = _groups(res)

    rows: list[dict[str, Any]] = []
    for (task, temp, gamma), meas in sorted(measured.items()):
        runs = groups[(task, temp)]
        base_runs = [r for r in runs if r["gamma"] == 0]
        spec_runs = [r for r in runs if r["gamma"] == gamma]

        # Host overhead per forward (sampling, Python, sync), from gamma=0 only.
        overhead = max(_per_token(base_runs) - lat.target(1), 0.0)
        lat_o = lat.with_overhead(overhead)

        # Cross-fit: predict each prompt's run from the other prompts' traces.
        pred_time = 0.0
        pred_tokens = 0
        alphas: list[float] = []
        for i, run in enumerate(spec_runs):
            cyc = _cycles(runs, exclude_gamma=gamma, exclude_prompt=run.get("prompt_id"))
            if not cyc:
                cyc = _cycles(runs, exclude_gamma=gamma)
            if not cyc:
                break
            pos = PositionalAcceptance.from_traces(cyc)
            alphas.append(pos.mean_alpha)
            acc: AcceptanceModel = (
                pos if acceptance == "positional" else IIDAcceptance(pos.mean_alpha)
            )
            n = max(2, int(run["n_new"]))
            sim = simulate(acc, gamma, lat_o, tokens=n, trials=trials, seed=seed + i)
            pred_time += sim.speculative_time
            pred_tokens += n
        if not alphas:
            continue
        pred = lat_o.target(1) * pred_tokens / pred_time

        own = _cycles(spec_runs)
        rows.append(
            {
                "task": task,
                "temperature": temp,
                "gamma": gamma,
                "measured": meas,
                "predicted": pred,
                "error_pct": 100.0 * (pred - meas) / meas,
                "alpha_fit": float(np.mean(alphas)),
                "host_overhead_ms": overhead * 1e3,
                "tokens_per_cycle_measured": (
                    sum(a + 1 for _, a in own) / len(own) if own else float("nan")
                ),
            }
        )

    errs = np.array([abs(r["error_pct"]) for r in rows]) if rows else np.array([np.nan])
    best = max(rows, key=lambda r: r["measured"]) if rows else None
    return {
        "meta": res["meta"],
        "acceptance_model": acceptance,
        "latency_context": ctx,
        "rows": rows,
        "summary": {
            "n": len(rows),
            "mae_pct": float(np.mean(errs)),
            "max_abs_error_pct": float(np.max(errs)),
            "best_measured": best["measured"] if best else float("nan"),
            "best_config": (
                {k: best[k] for k in ("task", "temperature", "gamma")} if best else None
            ),
        },
    }
