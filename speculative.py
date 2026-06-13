"""Speculative decoding simulator.

Speculative decoding is an optimisation technique that pairs a large
target model with a smaller draft model.  The draft model quickly
proposes a few tokens ahead, and the target model verifies them in
parallel.  If the draft tokens match the target model’s own
predictions, they are accepted; otherwise, the target recomputes
from the last accepted token.  This draft‑then‑verify process can
reduce the number of expensive target model forward passes and thus
accelerate inference【231741701474535†L60-L74】.  The key parameter is the
**acceptance rate** α, which denotes the probability that a draft
token is accepted by the target【231741701474535†L140-L149】.  Higher α values
produce greater speedups【231741701474535†L169-L170】.

This module simulates speculative decoding using a simple generative
model.  You can configure the acceptance rate, the number of draft
tokens proposed per cycle (γ), and the per‑token latencies for both
the draft and target models.  The simulator returns the total time
taken and the speedup relative to baseline decoding with the target
model only.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Optional


def run_speculative_simulation(
    max_tokens: int,
    accept_rate: float,
    draft_speed: float,
    target_speed: float,
    draft_batch_size: int = 4,
    seed: Optional[int] = None,
) -> dict:
    """Simulate speculative decoding.

    Parameters
    ----------
    max_tokens : int
        Total number of tokens to generate.
    accept_rate : float
        Probability that a draft token is accepted by the target (α).
    draft_speed : float
        Time (in arbitrary units) taken by the draft model to produce one token.
    target_speed : float
        Time taken by the target model to produce or verify one token.
    draft_batch_size : int, default 4
        Number of tokens proposed by the draft model each cycle (γ).
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing the total speculative time, baseline time and
        speedup (baseline_time / speculative_time).
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not (0.0 <= accept_rate <= 1.0):
        raise ValueError("accept_rate must be between 0 and 1")
    if draft_speed <= 0 or target_speed <= 0:
        raise ValueError("draft_speed and target_speed must be positive")
    if draft_batch_size <= 0:
        raise ValueError("draft_batch_size must be positive")
    rng = random.Random(seed)
    tokens_generated = 0
    time_elapsed = 0.0
    while tokens_generated < max_tokens:
        # Draft proposes up to draft_batch_size tokens ahead
        remaining = max_tokens - tokens_generated
        batch = min(draft_batch_size, remaining)
        # Draft model time cost
        time_elapsed += batch * draft_speed
        # Determine how many tokens are accepted in this batch
        accepted = 0
        for _ in range(batch):
            if tokens_generated + accepted >= max_tokens:
                break
            # Accept token with probability alpha until the first rejection
            if rng.random() < accept_rate:
                accepted += 1
            else:
                break
        # The target model verifies all proposed tokens in a single
        # forward pass and then generates the next token in another
        # forward pass【231741701474535†L115-L125】.  The cost does not depend on
        # how many tokens are accepted (verification and one new
        # generation), but we cannot exceed the total number of tokens.
        # Target pass 1: verify the batch; target pass 2: generate next token.
        # We still generate one additional token even if the batch is fully
        # accepted, to ensure the target stays ahead of the draft.
        # If we are near the end, we may not need both passes; clamp.
        remaining_tokens = max_tokens - tokens_generated
        # Each cycle processes accepted+1 tokens, but if we have fewer
        # remaining tokens, adjust the count.
        tokens_to_process = min(accepted + 1, remaining_tokens)
        # Verification cost (always one pass) plus generation cost (if we
        # still have tokens after this cycle)
        # We add the generation pass only if tokens_to_process > accepted
        # But since tokens_to_process = accepted + 1 unless at end, we
        # assume two passes when remaining_tokens >= 1.
        passes = 2 if remaining_tokens > 0 else 0
        time_elapsed += passes * target_speed
        tokens_generated += tokens_to_process
    # Baseline decoding: target model generates every token sequentially
    baseline_time = max_tokens * target_speed
    speedup = baseline_time / time_elapsed if time_elapsed > 0 else float("inf")
    return {
        "speculative_time": time_elapsed,
        "baseline_time": baseline_time,
        "speedup": speedup,
    }


def cli(argv: Optional[list[str]] = None) -> None:
    """Command‑line interface for running the speculative decoding simulator."""
    parser = argparse.ArgumentParser(description="Simulate speculative decoding speedups.")
    parser.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--accept-rate", type=float, default=0.6, help="Acceptance rate α")
    parser.add_argument("--draft-speed", type=float, default=0.1, help="Time per token for the draft model")
    parser.add_argument("--target-speed", type=float, default=0.2, help="Time per token for the target model")
    parser.add_argument("--draft-batch-size", type=int, default=4, help="Number of tokens proposed per cycle γ")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args(argv)
    result = run_speculative_simulation(
        max_tokens=args.tokens,
        accept_rate=args.accept_rate,
        draft_speed=args.draft_speed,
        target_speed=args.target_speed,
        draft_batch_size=args.draft_batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()