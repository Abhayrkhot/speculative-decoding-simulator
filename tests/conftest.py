"""Shared fixtures: a synthetic benchmark result with known ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from specsim.latency import LinearLatency

TRUE_LATENCY = LinearLatency(
    target_fixed_s=0.020, target_per_token_s=0.0004, draft_fixed_s=0.004, draft_per_token_s=0.0
)
TRUE_ALPHAS = {
    "code": [0.92, 0.88, 0.85, 0.83, 0.81, 0.8, 0.8, 0.8],
    "chat": [0.7, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5],
}
HOST_OVERHEAD_S = 0.001


def synthetic_results(seed: int = 0, gammas: tuple[int, ...] = (0, 1, 2, 4, 6)) -> dict[str, Any]:
    """Fake ``spec_bench`` output generated from a known latency/acceptance model."""
    rng = np.random.default_rng(seed)
    lat = TRUE_LATENCY
    ks = [1, 2, 3, 4, 5, 6, 8, 12, 16]
    profile = {
        "target": {str(k): lat.target(k) * 1e3 for k in ks},
        "draft": {str(k): lat.draft(k) * 1e3 for k in (1, 2, 4)},
    }
    runs = []
    max_new = 160
    for task, alphas in TRUE_ALPHAS.items():
        a = np.array(alphas)
        for gamma in gammas:
            for pid in range(6):
                produced, t, cycles = 0, 0.0, []
                while produced < max_new:
                    k = min(gamma, max_new - produced - 1)
                    hits = rng.random(k) < a[:k]
                    n_acc = int(np.cumprod(hits).sum()) if k else 0
                    t += k * (lat.draft(1) + HOST_OVERHEAD_S) + lat.target(k + 1) + HOST_OVERHEAD_S
                    produced += n_acc + 1
                    cycles.append([k, n_acc])
                runs.append(
                    {
                        "task": task,
                        "prompt_id": pid,
                        "prompt_tokens": 60,
                        "gamma": gamma,
                        "temperature": 0.0,
                        "n_new": max_new,
                        "wall_s": t + 0.05,
                        "prefill_s": 0.05,
                        "decode_s": t,
                        "cycles": cycles,
                        "tokens": [],
                    }
                )
    return {
        "meta": {
            "draft": "toy-draft",
            "target": "toy-target",
            "device": "synthetic",
            "max_new": max_new,
        },
        "latency_ms": {"256": profile},
        "runs": runs,
    }


@pytest.fixture(scope="session")
def bench_results() -> dict[str, Any]:
    return synthetic_results()


@pytest.fixture(scope="session")
def bench_file(tmp_path_factory: pytest.TempPathFactory, bench_results: dict[str, Any]) -> Path:
    p = tmp_path_factory.mktemp("bench") / "results.json"
    p.write_text(json.dumps(bench_results))
    return p
