"""Closed-form expectations for speculative decoding.

With conditional acceptance probabilities ``alpha_1..alpha_gamma`` the expected
number of tokens emitted per draft-verify cycle is

    E[A + 1] = 1 + sum_{i=1..gamma} prod_{j<=i} alpha_j

which for i.i.d. acceptance reduces to ``(1 - alpha**(gamma+1)) / (1 - alpha)``
(Leviathan et al., 2023, eq. 1). Dividing by the cycle cost gives the expected
speedup over plain decoding. These are infinite-horizon expectations; the Monte
Carlo engine in :mod:`specsim.simulate` converges to them as the generation
length grows.
"""

from __future__ import annotations

from collections.abc import Sequence

from .acceptance import AcceptanceModel, as_acceptance
from .latency import ConstantLatency, LatencyModel

__all__ = [
    "expected_cycle_time",
    "expected_speedup",
    "expected_tokens_per_cycle",
    "leviathan_speedup",
    "optimal_gamma",
]

AcceptanceLike = AcceptanceModel | float | Sequence[float]


def _check_gamma(gamma: int) -> None:
    if gamma < 0 or int(gamma) != gamma:
        raise ValueError(f"gamma must be a non-negative integer, got {gamma!r}")


def expected_tokens_per_cycle(acceptance: AcceptanceLike, gamma: int) -> float:
    """Expected tokens emitted by one cycle with ``gamma`` draft tokens."""
    _check_gamma(gamma)
    total, run = 1.0, 1.0
    for a in as_acceptance(acceptance).alphas(gamma):
        run *= float(a)
        total += run
    return total


def expected_cycle_time(gamma: int, latency: LatencyModel, batch: int = 1) -> float:
    """Draft ``gamma`` tokens sequentially, then verify them in one target pass."""
    _check_gamma(gamma)
    return gamma * latency.draft(1, batch) + latency.target(gamma + 1, batch)


def expected_speedup(
    acceptance: AcceptanceLike, gamma: int, latency: LatencyModel, batch: int = 1
) -> float:
    """Expected throughput ratio of speculative over plain target decoding."""
    if gamma == 0:
        return 1.0
    per_token_spec = expected_cycle_time(gamma, latency, batch) / expected_tokens_per_cycle(
        acceptance, gamma
    )
    return latency.target(1, batch) / per_token_spec


def leviathan_speedup(alpha: float, gamma: int, c: float) -> float:
    """Leviathan et al. (2023) Thm 3.8: ``(1 - a^(g+1)) / ((1 - a)(g c + 1))``."""
    return expected_speedup(alpha, gamma, ConstantLatency.from_ratio(c))


def optimal_gamma(
    acceptance: AcceptanceLike, latency: LatencyModel, max_gamma: int = 32, batch: int = 1
) -> tuple[int, float]:
    """Return ``(gamma, speedup)`` maximising expected speedup.

    ``gamma=0`` (speedup 1.0) means speculation does not pay off at all.
    """
    _check_gamma(max_gamma)
    acc = as_acceptance(acceptance)
    best = (0, 1.0)
    for g in range(1, max_gamma + 1):
        s = expected_speedup(acc, g, latency, batch)
        if s > best[1] + 1e-12:
            best = (g, s)
    return best
