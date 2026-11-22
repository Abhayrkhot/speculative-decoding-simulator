"""Exact speculative sampling on explicit distributions.

The Monte Carlo engine in :mod:`specsim.simulate` treats acceptance as a coin
with probability ``alpha``. This module implements the actual algorithm
(Leviathan et al., 2023, Alg. 1; Chen et al., 2023) on a toy language model so
that ``alpha`` *emerges* from how close the draft is to the target, and so that
the central correctness claim - speculative sampling reproduces the target
model's output distribution exactly - can be tested statistically.

The toy model is a first-order Markov chain over a small vocabulary: the target
has transition matrix ``P`` and the draft ``Q``, built from the same random
logits plus noise of scale ``divergence``. For a context ``c`` the probability
of accepting a draft token is ``sum_x min(P[c, x], Q[c, x]) = 1 - TV(P[c], Q[c])``.

Algorithm, for each draft position ``i`` with context ``c_i``:

1. draft samples ``x_i ~ Q[c_i]``;
2. target accepts with probability ``min(1, P[c_i, x_i] / Q[c_i, x_i])``;
3. on the first rejection, emit a sample from ``norm(max(0, P[c_i] - Q[c_i]))``
   and end the cycle; if all ``gamma`` drafts are accepted, emit a bonus token
   from ``P[x_gamma]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MarkovPair",
    "acceptance_probability",
    "chi_square_sf",
    "lossless_check",
    "sample_target",
    "speculative_sample",
]


def acceptance_probability(p: np.ndarray, q: np.ndarray) -> float:
    """``sum_x min(p(x), q(x))`` = 1 - total-variation distance."""
    return float(np.minimum(p, q).sum())


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    out: np.ndarray = e / e.sum(axis=-1, keepdims=True)
    return out


def _sample_rows(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one index per row of a (n, V) probability matrix (inverse CDF)."""
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0]) * cdf[:, -1]
    idx: np.ndarray = (cdf < u[:, None]).sum(axis=1)
    out: np.ndarray = np.minimum(idx, probs.shape[1] - 1)
    return out


@dataclass(frozen=True)
class MarkovPair:
    """Target/draft bigram models over a vocabulary of size ``V``."""

    P: np.ndarray  # (V, V) target transitions
    Q: np.ndarray  # (V, V) draft transitions

    @classmethod
    def random(
        cls, vocab: int, divergence: float, sharpness: float = 2.0, seed: int | None = None
    ) -> MarkovPair:
        """Random target logits ``N(0, sharpness^2)``; draft adds ``N(0, divergence^2)`` noise."""
        if vocab < 2:
            raise ValueError("vocab must be at least 2")
        if divergence < 0:
            raise ValueError("divergence must be non-negative")
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal((vocab, vocab)) * sharpness
        noisy = logits + rng.standard_normal((vocab, vocab)) * divergence
        return cls(_softmax(logits), _softmax(noisy))

    @property
    def vocab(self) -> int:
        return int(self.P.shape[0])

    def row_acceptance(self) -> np.ndarray:
        """Acceptance probability for each context: ``1 - TV(P[c], Q[c])``."""
        out: np.ndarray = np.minimum(self.P, self.Q).sum(axis=1)
        return out

    def marginals(self, length: int, start: int = 0) -> np.ndarray:
        """Exact distribution of the target's token at positions ``1..length``."""
        dist = np.zeros(self.vocab)
        dist[start] = 1.0
        rows = []
        for _ in range(length):
            dist = dist @ self.P
            rows.append(dist)
        return np.array(rows)


def sample_target(
    pair: MarkovPair, n: int, length: int, rng: np.random.Generator, start: int = 0
) -> np.ndarray:
    """Plain ancestral sampling from the target: ``(n, length)`` tokens."""
    out = np.empty((n, length), dtype=np.int64)
    last = np.full(n, start)
    for t in range(length):
        last = _sample_rows(pair.P[last], rng)
        out[:, t] = last
    return out


def speculative_sample(
    pair: MarkovPair,
    n: int,
    length: int,
    gamma: int,
    rng: np.random.Generator,
    start: int = 0,
    residual: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate ``n`` sequences of ``length`` tokens with speculative sampling.

    ``residual=False`` replaces the residual distribution with ``P`` on
    rejection - a common implementation bug that biases the output towards the
    draft. It exists so tests can show the statistical check catches it.

    Returns the ``(n, length)`` token matrix and acceptance statistics.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    P, Q = pair.P, pair.Q  # noqa: N806 - matrix notation
    buf = np.full((n, length + gamma + 1), -1, dtype=np.int64)
    pos = np.zeros(n, dtype=np.int64)
    last = np.full(n, start, dtype=np.int64)
    proposed = accepted = n_emitted = 0
    expected_acc = 0.0
    cycles = 0
    row_acc = pair.row_acceptance()

    active = np.arange(n)
    while active.size:
        m = active.size
        ctx = np.empty((m, gamma), dtype=np.int64)
        drafts = np.empty((m, gamma), dtype=np.int64)
        c = last[active]
        for i in range(gamma):
            ctx[:, i] = c
            c = _sample_rows(Q[c], rng)
            drafts[:, i] = c

        p_x = P[ctx, drafts]
        q_x = Q[ctx, drafts]
        hits = rng.random((m, gamma)) * q_x <= p_x
        n_acc = np.cumprod(hits, axis=1).sum(axis=1)

        # Correction token at the first rejected slot, or bonus token after all-accept.
        extra = np.empty(m, dtype=np.int64)
        rej = n_acc < gamma
        if rej.any():
            rc = ctx[rej, n_acc[rej]]
            if residual:
                dist = np.maximum(P[rc] - Q[rc], 0.0)
                zero = dist.sum(axis=1) <= 0
                dist[zero] = P[rc[zero]]
            else:
                dist = P[rc]
            extra[rej] = _sample_rows(dist, rng)
        if (~rej).any():
            extra[~rej] = _sample_rows(P[drafts[~rej, -1]], rng)

        # Write emitted tokens: drafts[:, :n_acc] then extra.
        slots = np.arange(gamma + 1)
        emitted = np.where(slots[None, :] < n_acc[:, None], np.pad(drafts, ((0, 0), (0, 1))), -1)
        emitted[np.arange(m), n_acc] = extra
        cols = pos[active, None] + slots[None, :]
        valid = slots[None, :] <= n_acc[:, None]
        rows = np.broadcast_to(active[:, None], cols.shape)
        buf[rows[valid], cols[valid]] = emitted[valid]

        # Statistics only over slots that were actually verified.
        tried = slots[None, :gamma] <= np.minimum(n_acc, gamma - 1)[:, None]
        proposed += int(tried.sum())
        accepted += int(n_acc.sum())
        n_emitted += int((n_acc + 1).sum())
        expected_acc += float(row_acc[ctx][tried].sum())
        cycles += m

        pos[active] += n_acc + 1
        last[active] = extra
        active = active[pos[active] < length]

    stats = {
        "acceptance_rate": accepted / proposed if proposed else float("nan"),
        "expected_acceptance": expected_acc / proposed if proposed else float("nan"),
        "tokens_per_cycle": n_emitted / cycles,
    }
    return buf[:, :length], stats


def chi_square_sf(stat: float, dof: int) -> float:
    """Survival function of the chi-square distribution (Wilson-Hilferty approximation)."""
    if dof <= 0:
        return 1.0
    k = float(dof)
    z = ((stat / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _chi_square(
    observed: np.ndarray, expected: np.ndarray, min_expected: float = 5.0
) -> tuple[float, int]:
    """Pearson statistic, pooling cells with small expected counts into one bin."""
    small = expected < min_expected
    obs = np.append(observed[~small], observed[small].sum())
    exp = np.append(expected[~small], expected[small].sum())
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    stat = float(((obs - exp) ** 2 / exp).sum())
    return stat, max(int(keep.sum()) - 1, 0)


def lossless_check(
    vocab: int = 32,
    gamma: int = 4,
    divergence: float = 1.0,
    samples: int = 200_000,
    length: int = 6,
    seed: int = 0,
    residual: bool = True,
    alpha_level: float = 1e-3,
) -> dict[str, object]:
    """Statistically test that speculative sampling reproduces the target exactly.

    Two families of chi-square goodness-of-fit tests against *exact* target
    quantities (no sampling noise on the reference side):

    * the marginal distribution of the token at every position ``1..length``
      versus ``e_start @ P^t``;
    * pooled bigram transitions versus ``P`` (tests dependence across cycle
      boundaries, not just marginals).

    Passes if every Bonferroni-corrected p-value exceeds ``alpha_level``.
    """
    model_seed, sample_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(sample_seed)
    pair = MarkovPair.random(vocab, divergence, seed=int(model_seed.generate_state(1)[0]))
    seqs, stats = speculative_sample(pair, samples, length, gamma, rng, residual=residual)

    marg = pair.marginals(length)
    p_values: list[float] = []
    for t in range(length):
        obs = np.bincount(seqs[:, t], minlength=vocab).astype(float)
        stat, dof = _chi_square(obs, marg[t] * samples)
        p_values.append(chi_square_sf(stat, dof))

    prev = np.concatenate([np.zeros((samples, 1), dtype=np.int64), seqs[:, :-1]], axis=1)
    pairs = np.zeros((vocab, vocab))
    np.add.at(pairs, (prev.ravel(), seqs.ravel()), 1.0)
    # Expected counts use each row's observed total, so each non-empty row
    # contributes (bins - 1) degrees of freedom.
    expected = pairs.sum(axis=1, keepdims=True) * pair.P
    stat, dof = 0.0, 0
    for a in range(vocab):
        if pairs[a].sum() == 0:
            continue
        s_a, d_a = _chi_square(pairs[a], expected[a])
        stat += s_a
        dof += d_a
    p_bigram = chi_square_sf(stat, dof)

    n_tests = length + 1
    min_p = min([*p_values, p_bigram])
    return {
        "vocab": vocab,
        "gamma": gamma,
        "divergence": divergence,
        "samples": samples,
        "residual_correction": residual,
        "acceptance_rate": stats["acceptance_rate"],
        "expected_acceptance": stats["expected_acceptance"],
        "tokens_per_cycle": stats["tokens_per_cycle"],
        "marginal_p_values": p_values,
        "bigram_p_value": p_bigram,
        "min_p_bonferroni": min(1.0, min_p * n_tests),
        "passed": bool(min_p * n_tests > alpha_level),
    }
