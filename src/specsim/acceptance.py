"""Acceptance models: how likely the target is to accept each draft token.

The quantity that matters is the *conditional* acceptance probability at each
draft position: ``alpha_i = P(token i accepted | tokens 1..i-1 accepted)``.
Verification stops at the first rejection, so the number of accepted tokens in
a cycle is ``A = #leading accepts`` and the cycle emits ``A + 1`` tokens (the
target always contributes one: the correction or the bonus token).

``IIDAcceptance``
    Every position has the same ``alpha`` - the textbook assumption.
``PositionalAcceptance``
    One ``alpha`` per position, typically estimated from measured traces with
    :meth:`PositionalAcceptance.from_traces`. Real drafts get less reliable the
    further ahead they guess, which the i.i.d. model cannot express.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["AcceptanceModel", "IIDAcceptance", "PositionalAcceptance", "as_acceptance"]


@runtime_checkable
class AcceptanceModel(Protocol):
    def alphas(self, gamma: int) -> np.ndarray:
        """Conditional acceptance probability for draft positions ``1..gamma``."""
        ...


def _check_prob(name: str, p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p!r}")


@dataclass(frozen=True)
class IIDAcceptance:
    alpha: float

    def __post_init__(self) -> None:
        _check_prob("alpha", self.alpha)

    def alphas(self, gamma: int) -> np.ndarray:
        return np.full(gamma, self.alpha, dtype=float)


@dataclass(frozen=True)
class PositionalAcceptance:
    """Per-position conditional acceptance; positions past the end reuse the last value."""

    values: tuple[float, ...]
    counts: tuple[int, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("need at least one acceptance probability")
        for i, a in enumerate(self.values):
            _check_prob(f"values[{i}]", a)

    def alphas(self, gamma: int) -> np.ndarray:
        v = np.asarray(self.values, dtype=float)
        out: np.ndarray = (
            v[:gamma].copy()
            if gamma <= len(v)
            else np.concatenate([v, np.full(gamma - len(v), v[-1])])
        )
        return out

    @property
    def mean_alpha(self) -> float:
        """Pooled acceptance rate (weighted by how often each position was tried)."""
        if not self.counts:
            return float(np.mean(self.values))
        w = np.asarray(self.counts, dtype=float)
        return float(np.dot(self.values, w) / w.sum())

    @classmethod
    def from_traces(
        cls, cycles: Iterable[Sequence[int]], min_count: int = 30
    ) -> PositionalAcceptance:
        """Estimate from ``(k, n_accepted)`` pairs, one per draft-verify cycle.

        Position ``i`` (0-based) was *tried* in a cycle iff ``k > i`` and the
        first ``i`` drafts were accepted (``n_accepted >= i``); it *succeeded*
        iff ``n_accepted > i``. Positions tried fewer than ``min_count`` times
        are dropped (they inherit the last well-estimated value).
        """
        arr = np.asarray([(int(k), int(a)) for k, a in cycles], dtype=int).reshape(-1, 2)
        if arr.size == 0:
            raise ValueError("no cycles to estimate from")
        k, a = arr[:, 0], arr[:, 1]
        if np.any(a > k) or np.any(a < 0):
            raise ValueError("each cycle needs 0 <= n_accepted <= k")
        values: list[float] = []
        counts: list[int] = []
        for i in range(int(k.max())):
            tried = (k > i) & (a >= i)
            n = int(tried.sum())
            if n < min_count:
                break
            values.append(float((a[tried] > i).mean()))
            counts.append(n)
        if not values:
            # Too little data per position: fall back to the pooled MLE.
            tried_total = int(np.minimum(a + 1, k).sum())
            if tried_total == 0:
                raise ValueError("no draft tokens were proposed in these cycles")
            values, counts = [float(a.sum() / tried_total)], [tried_total]
        return cls(tuple(values), tuple(counts))


def as_acceptance(model: AcceptanceModel | float | Sequence[float]) -> AcceptanceModel:
    """Coerce a float (i.i.d.) or a sequence (positional) into an acceptance model."""
    if isinstance(model, (int, float)):
        return IIDAcceptance(float(model))
    if isinstance(model, AcceptanceModel):
        return model
    return PositionalAcceptance(tuple(float(x) for x in model))
