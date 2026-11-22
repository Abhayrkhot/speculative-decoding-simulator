"""Vectorised Monte Carlo simulation of speculative decoding.

Simulates ``trials`` independent generations of ``tokens`` tokens each. A cycle
proposes ``k = min(gamma, remaining - 1)`` draft tokens (so a generation never
overshoots its length), verifies them with one target forward pass, and emits
``accepted + 1`` tokens. This is exactly the loop in ``bench/spec_bench.py``,
so simulated and measured numbers are directly comparable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .acceptance import AcceptanceModel, as_acceptance
from .latency import LatencyModel

__all__ = ["SimulationResult", "simulate"]


@dataclass(frozen=True)
class SimulationResult:
    """Summary statistics of a Monte Carlo run. Times are in the latency model's units."""

    gamma: int
    tokens: int
    trials: int
    batch: int
    speedup: float
    """Throughput ratio over the whole workload: total baseline time / total speculative time."""
    speedup_ci95: tuple[float, float]
    """95% confidence interval for ``speedup`` (delta method)."""
    speedup_p05: float
    """5th percentile of per-generation speedup."""
    speedup_p95: float
    """95th percentile of per-generation speedup."""
    speculative_time: float
    """Mean wall time of one generation with speculative decoding."""
    baseline_time: float
    """Wall time of one generation with plain target decoding."""
    tokens_per_cycle: float
    """Mean tokens emitted per draft-verify cycle."""
    acceptance_rate: float
    """Accepted / verified draft positions (estimates alpha; stops at first rejection)."""
    draft_efficiency: float
    """Accepted / drafted tokens (drafts after a rejection are wasted work)."""
    target_calls_saved: float
    """Fraction of target forward passes avoided relative to the baseline."""

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["speedup_ci95"] = list(self.speedup_ci95)
        return d


def simulate(
    acceptance: AcceptanceModel | float | Sequence[float],
    gamma: int,
    latency: LatencyModel,
    tokens: int = 256,
    trials: int = 1000,
    batch: int = 1,
    seed: int | None = None,
) -> SimulationResult:
    """Run the Monte Carlo simulation.

    Parameters
    ----------
    acceptance
        An :class:`~specsim.acceptance.AcceptanceModel`, a float (i.i.d. alpha)
        or a sequence of per-position conditional acceptance probabilities.
    gamma
        Draft tokens proposed per cycle (``0`` = plain decoding).
    latency
        Forward-pass latency model.
    tokens
        Tokens generated per sequence.
    trials
        Independent generations to simulate.
    batch
        Sequences decoded together. Acceptance is simulated per trial; the
        latency model is evaluated at this batch size.
    seed
        Seed for :func:`numpy.random.default_rng`.
    """
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens!r}")
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials!r}")
    if gamma < 0 or int(gamma) != gamma:
        raise ValueError(f"gamma must be a non-negative integer, got {gamma!r}")
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch!r}")

    rng = np.random.default_rng(seed)
    alphas = as_acceptance(acceptance).alphas(gamma)
    t_draft = latency.draft(1, batch)
    t_verify = np.array([latency.target(k + 1, batch) for k in range(gamma + 1)])
    positions = np.arange(gamma)

    produced = np.zeros(trials, dtype=np.int64)
    elapsed = np.zeros(trials)
    cycles = np.zeros(trials, dtype=np.int64)
    accepted = np.zeros(trials, dtype=np.int64)
    proposed = np.zeros(trials, dtype=np.int64)
    verified = np.zeros(trials, dtype=np.int64)

    active = np.arange(trials)
    while active.size:
        k = np.minimum(gamma, tokens - produced[active] - 1)
        hits = rng.random((active.size, gamma)) < alphas
        hits &= positions < k[:, None]
        n_acc = np.cumprod(hits, axis=1).sum(axis=1)

        produced[active] += n_acc + 1
        elapsed[active] += k * t_draft + t_verify[k]
        cycles[active] += 1
        accepted[active] += n_acc
        proposed[active] += k
        verified[active] += np.minimum(n_acc + 1, k)
        active = active[produced[active] < tokens]

    baseline = tokens * latency.target(1, batch)
    mean_t = float(elapsed.mean())
    se_t = float(elapsed.std(ddof=1)) / math.sqrt(trials) if trials > 1 else 0.0
    speedup = baseline / mean_t
    half = 1.96 * baseline * se_t / mean_t**2
    per_gen = baseline / elapsed
    total_proposed = int(proposed.sum())
    total_verified = int(verified.sum())
    return SimulationResult(
        gamma=gamma,
        tokens=tokens,
        trials=trials,
        batch=batch,
        speedup=speedup,
        speedup_ci95=(speedup - half, speedup + half),
        speedup_p05=float(np.percentile(per_gen, 5)),
        speedup_p95=float(np.percentile(per_gen, 95)),
        speculative_time=mean_t,
        baseline_time=baseline,
        tokens_per_cycle=int(produced.sum()) / int(cycles.sum()),
        acceptance_rate=int(accepted.sum()) / total_verified if total_verified else float("nan"),
        draft_efficiency=int(accepted.sum()) / total_proposed if total_proposed else float("nan"),
        target_calls_saved=1.0 - float(cycles.mean()) / tokens,
    )
