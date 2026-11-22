"""specsim: a calibrated simulator for speculative decoding.

Quick start::

    >>> from specsim import simulate, ConstantLatency
    >>> r = simulate(0.8, gamma=4, latency=ConstantLatency.from_ratio(0.05), seed=0)
    >>> round(r.speedup, 1)
    2.8
"""

from __future__ import annotations

__version__ = "0.2.0"

from .acceptance import AcceptanceModel, IIDAcceptance, PositionalAcceptance
from .analytic import (
    expected_cycle_time,
    expected_speedup,
    expected_tokens_per_cycle,
    leviathan_speedup,
    optimal_gamma,
)
from .latency import (
    HARDWARE,
    MODELS,
    ConstantLatency,
    Hardware,
    LatencyModel,
    LinearLatency,
    ModelSpec,
    RooflineLatency,
    TabulatedLatency,
)
from .simulate import SimulationResult, simulate

__all__ = [
    "HARDWARE",
    "MODELS",
    "AcceptanceModel",
    "ConstantLatency",
    "Hardware",
    "IIDAcceptance",
    "LatencyModel",
    "LinearLatency",
    "ModelSpec",
    "PositionalAcceptance",
    "RooflineLatency",
    "SimulationResult",
    "TabulatedLatency",
    "__version__",
    "expected_cycle_time",
    "expected_speedup",
    "expected_tokens_per_cycle",
    "leviathan_speedup",
    "optimal_gamma",
    "simulate",
]
