import math

import os
import sys

# Add the parent directory to sys.path to import speculative.py when
# running tests from the repository root.  Without this modification
# Python will not recognise the module because the project directory
# is not installed as a package.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speculative import run_speculative_simulation


def test_speedup_extremes():
    """Speculative decoding should provide speedup for high acceptance and slow down for zero acceptance."""
    # Baseline: target_speed=0.2, 50 tokens -> baseline_time = 10.0
    baseline_time = 50 * 0.2
    # High acceptance: accept_rate=1.0 should accelerate relative to baseline
    result_high = run_speculative_simulation(
        max_tokens=50,
        accept_rate=1.0,
        draft_speed=0.05,
        target_speed=0.2,
        draft_batch_size=4,
        seed=42,
    )
    assert result_high["baseline_time"] == baseline_time
    assert result_high["speedup"] > 1.0
    # Zero acceptance: accept_rate=0.0 wastes time generating draft tokens that are never accepted
    result_low = run_speculative_simulation(
        max_tokens=50,
        accept_rate=0.0,
        draft_speed=0.05,
        target_speed=0.2,
        draft_batch_size=4,
        seed=42,
    )
    assert result_low["baseline_time"] == baseline_time
    # Speedup should be less than 1 (i.e. slower than baseline)
    assert result_low["speedup"] < 1.0