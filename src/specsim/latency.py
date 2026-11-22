"""Latency models: how long one forward pass of the draft or target model takes.

Every model answers two questions for a batch of ``batch`` sequences that each
append ``k`` new tokens to their KV cache in one forward pass:

* ``target(k, batch)`` - seconds for one target-model forward pass;
* ``draft(k, batch)``  - seconds for one draft-model forward pass.

A speculative decoding cycle with ``gamma`` draft tokens costs
``gamma * draft(1, B) + target(gamma + 1, B)``; plain decoding costs
``target(1, B)`` per token.

Four models are provided, from simplest to most physical:

``ConstantLatency``
    A forward pass costs the same regardless of how many tokens it processes.
    This is the memory-bound idealisation used by Leviathan et al. (2023), where
    the only parameter that matters is the cost ratio ``c = draft / target``.
``LinearLatency``
    ``t(k) = fixed + per_token * k * batch``. Fitted from measured profiles
    (see :mod:`specsim.calibrate`); captures kernel-launch overhead and the
    growing cost of verifying more tokens.
``TabulatedLatency``
    Interpolates a measured profile directly; used for validation.
``RooflineLatency``
    ``t = overhead + max(bytes / bandwidth, flops / peak)`` from model size,
    KV-cache size and hardware specs. Shows the transition from memory-bound to
    compute-bound decoding as batch size grows - the regime in which
    speculative decoding stops paying off.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "HARDWARE",
    "MODELS",
    "ConstantLatency",
    "Hardware",
    "LatencyModel",
    "LinearLatency",
    "ModelSpec",
    "RooflineLatency",
    "TabulatedLatency",
]


@runtime_checkable
class LatencyModel(Protocol):
    """Protocol for forward-pass latency models (all times in seconds)."""

    def target(self, k: int, batch: int = 1) -> float: ...

    def draft(self, k: int, batch: int = 1) -> float: ...


def _positive(name: str, value: float) -> None:
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _non_negative(name: str, value: float) -> None:
    if not value >= 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


@dataclass(frozen=True)
class ConstantLatency:
    """Forward-pass cost independent of the number of tokens processed."""

    target_s: float
    draft_s: float

    def __post_init__(self) -> None:
        _positive("target_s", self.target_s)
        _positive("draft_s", self.draft_s)

    @classmethod
    def from_ratio(cls, c: float, target_s: float = 1.0) -> ConstantLatency:
        """Build from the draft/target cost ratio ``c``."""
        return cls(target_s=target_s, draft_s=c * target_s)

    @property
    def c(self) -> float:
        """Draft/target cost ratio."""
        return self.draft_s / self.target_s

    def target(self, k: int, batch: int = 1) -> float:
        return self.target_s

    def draft(self, k: int, batch: int = 1) -> float:
        return self.draft_s


@dataclass(frozen=True)
class LinearLatency:
    """``t(k, batch) = fixed + per_token * k * batch`` for each model."""

    target_fixed_s: float
    target_per_token_s: float
    draft_fixed_s: float
    draft_per_token_s: float = 0.0

    def __post_init__(self) -> None:
        _positive("target_fixed_s", self.target_fixed_s)
        _positive("draft_fixed_s", self.draft_fixed_s)
        _non_negative("target_per_token_s", self.target_per_token_s)
        _non_negative("draft_per_token_s", self.draft_per_token_s)

    def target(self, k: int, batch: int = 1) -> float:
        return self.target_fixed_s + self.target_per_token_s * k * batch

    def draft(self, k: int, batch: int = 1) -> float:
        return self.draft_fixed_s + self.draft_per_token_s * k * batch

    @staticmethod
    def _fit_line(profile: Mapping[int, float]) -> tuple[float, float]:
        if not profile:
            raise ValueError("empty latency profile")
        ks = np.array(sorted(profile), dtype=float)
        ts = np.array([profile[int(k)] for k in ks], dtype=float)
        if len(ks) == 1:
            return float(ts[0]), 0.0
        slope, intercept = np.polyfit(ks, ts, 1)
        slope = max(float(slope), 0.0)
        # Refit the intercept with the (possibly clamped) slope.
        intercept = float(np.mean(ts - slope * ks))
        return max(intercept, 1e-9), slope

    @classmethod
    def fit(
        cls, target_profile: Mapping[int, float], draft_profile: Mapping[int, float]
    ) -> LinearLatency:
        """Least-squares fit from ``{k: seconds}`` profiles (batch size 1)."""
        tf, tp = cls._fit_line(target_profile)
        df, dp = cls._fit_line(draft_profile)
        return cls(target_fixed_s=tf, target_per_token_s=tp, draft_fixed_s=df, draft_per_token_s=dp)


def _interp(profile: tuple[tuple[int, float], ...], n: float) -> float:
    ks = np.array([k for k, _ in profile], dtype=float)
    ts = np.array([t for _, t in profile], dtype=float)
    if len(ks) == 1:
        return float(ts[0])
    if n <= ks[-1]:
        return float(np.interp(n, ks, ts))
    slope = max((ts[-1] - ts[-2]) / (ks[-1] - ks[-2]), 0.0)
    return float(ts[-1] + slope * (n - ks[-1]))


@dataclass(frozen=True)
class TabulatedLatency:
    """Piecewise-linear interpolation of a measured ``{k: seconds}`` profile.

    The most faithful model when a profile is available: it reproduces the
    measured cost of verifying ``k`` tokens exactly, including non-linear steps
    from kernel tile sizes. Beyond the largest measured ``k`` it extrapolates
    linearly. ``batch > 1`` is approximated as ``k * batch`` tokens.
    """

    target_profile: tuple[tuple[int, float], ...]
    draft_profile: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        for name in ("target_profile", "draft_profile"):
            prof = getattr(self, name)
            if not prof:
                raise ValueError(f"{name} is empty")
            if any(t <= 0 for _, t in prof):
                raise ValueError(f"{name} times must be positive")

    @classmethod
    def from_profiles(
        cls, target: Mapping[int, float], draft: Mapping[int, float]
    ) -> TabulatedLatency:
        return cls(
            tuple(sorted((int(k), float(v)) for k, v in target.items())),
            tuple(sorted((int(k), float(v)) for k, v in draft.items())),
        )

    def with_overhead(self, seconds: float) -> TabulatedLatency:
        """Add a fixed per-forward cost (e.g. host-side sampling) to both models."""
        return TabulatedLatency(
            tuple((k, t + seconds) for k, t in self.target_profile),
            tuple((k, t + seconds) for k, t in self.draft_profile),
        )

    def target(self, k: int, batch: int = 1) -> float:
        return _interp(self.target_profile, k * batch)

    def draft(self, k: int, batch: int = 1) -> float:
        return _interp(self.draft_profile, k * batch)


@dataclass(frozen=True)
class Hardware:
    """Accelerator specs. ``mem_bw`` in bytes/s, ``peak_flops`` in FLOP/s (dense bf16)."""

    name: str
    mem_bw: float
    peak_flops: float


@dataclass(frozen=True)
class ModelSpec:
    """Transformer size parameters relevant to decode cost."""

    name: str
    params: float
    n_layers: int
    kv_dim: int  # n_kv_heads * head_dim
    bytes_per_param: float = 2.0

    @property
    def weight_bytes(self) -> float:
        return self.params * self.bytes_per_param

    @property
    def kv_bytes_per_token(self) -> float:
        # K and V, every layer.
        return 2.0 * self.n_layers * self.kv_dim * self.bytes_per_param


HARDWARE: dict[str, Hardware] = {
    "a100-80gb": Hardware("NVIDIA A100 80GB SXM", 2.04e12, 312e12),
    "h100-sxm": Hardware("NVIDIA H100 SXM", 3.35e12, 989e12),
    "h200": Hardware("NVIDIA H200", 4.8e12, 989e12),
    "l4": Hardware("NVIDIA L4", 0.3e12, 121e12),
    "rtx-4090": Hardware("NVIDIA RTX 4090", 1.008e12, 165e12),
}

MODELS: dict[str, ModelSpec] = {
    "qwen2.5-0.5b": ModelSpec("Qwen2.5-0.5B", 0.49e9, 24, 2 * 64),
    "qwen2.5-1.5b": ModelSpec("Qwen2.5-1.5B", 1.54e9, 28, 2 * 128),
    "qwen2.5-7b": ModelSpec("Qwen2.5-7B", 7.62e9, 28, 4 * 128),
    "qwen2.5-32b": ModelSpec("Qwen2.5-32B", 32.8e9, 64, 8 * 128),
    "llama-3.2-1b": ModelSpec("Llama-3.2-1B", 1.24e9, 16, 8 * 64),
    "llama-3.1-8b": ModelSpec("Llama-3.1-8B", 8.03e9, 32, 8 * 128),
    "llama-3.1-70b": ModelSpec("Llama-3.1-70B", 70.6e9, 80, 8 * 128),
}


@dataclass(frozen=True)
class RooflineLatency:
    """Roofline estimate: ``overhead + max(bytes / bw, flops / peak)``.

    Bytes moved per forward = weights + KV cache of every sequence in the batch;
    FLOPs = ``2 * params * tokens``. ``bw_efficiency`` and ``flops_efficiency``
    are the achievable fractions of peak (memory-bandwidth utilisation and model
    FLOPs utilisation). ``*_overhead_s`` models fixed per-forward costs such as
    kernel launches and host synchronisation, which dominate small models in
    eager-mode frameworks.
    """

    target_model: ModelSpec
    draft_model: ModelSpec
    hardware: Hardware
    context: int = 1024
    bw_efficiency: float = 0.8
    flops_efficiency: float = 0.6
    target_overhead_s: float = 0.0
    draft_overhead_s: float = 0.0

    def __post_init__(self) -> None:
        _non_negative("context", self.context)
        for name in ("bw_efficiency", "flops_efficiency"):
            v = getattr(self, name)
            if not 0 < v <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {v!r}")
        _non_negative("target_overhead_s", self.target_overhead_s)
        _non_negative("draft_overhead_s", self.draft_overhead_s)

    def _time(self, m: ModelSpec, k: int, batch: int, overhead: float) -> float:
        bytes_moved = m.weight_bytes + batch * self.context * m.kv_bytes_per_token
        flops = 2.0 * m.params * k * batch
        t_mem = bytes_moved / (self.hardware.mem_bw * self.bw_efficiency)
        t_compute = flops / (self.hardware.peak_flops * self.flops_efficiency)
        return overhead + max(t_mem, t_compute)

    def target(self, k: int, batch: int = 1) -> float:
        return self._time(self.target_model, k, batch, self.target_overhead_s)

    def draft(self, k: int, batch: int = 1) -> float:
        return self._time(self.draft_model, k, batch, self.draft_overhead_s)

    def critical_tokens(self, m: ModelSpec | None = None) -> float:
        """Tokens per forward (k * batch) at which the target becomes compute-bound."""
        m = m or self.target_model
        bytes_moved = m.weight_bytes  # ignoring KV for the headline number
        return (bytes_moved / (self.hardware.mem_bw * self.bw_efficiency)) / (
            2.0 * m.params / (self.hardware.peak_flops * self.flops_efficiency)
        )
