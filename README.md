# Speculative Decoding Simulator

## Overview

Speculative decoding is an inference‑time optimisation that pairs a
large **target model** with a smaller **draft model**【231741701474535†L60-L66】.  The
draft model proposes several tokens ahead, and the target model
verifies them in parallel.  If the proposed tokens match the target’s
own predictions, they are accepted; otherwise, the target recomputes
the output from the last accepted token【231741701474535†L115-L125】.  This
draft‑then‑verify pattern reduces the number of expensive target model
forward passes, leading to up to 3× faster inference【231741701474535†L60-L74】.

This project implements a **speculative decoding simulator**.  It
models the acceptance of draft tokens with a probability α (the
**acceptance rate**)【231741701474535†L140-L149】 and measures the time spent
drafting and verifying tokens.  By comparing the speculative time to
a baseline where the target model generates every token, you can
explore how acceptance rate, draft size (γ) and per‑token latencies
affect speedup.  When α is high, speculative decoding achieves
significant speedups【231741701474535†L169-L170】.  When α is low, the
overhead of drafting wasted tokens can outweigh the benefits and
slow down inference.

## Why this matters

Transformers generate text autoregressively, producing one token at a
time【231741701474535†L76-L88】.  Each forward pass through a large model can be
expensive, and GPU compute often sits idle while waiting on memory
access【231741701474535†L101-L103】.  Speculative decoding parallelises part of
this process by letting a lightweight draft model propose multiple
tokens ahead and verifying them in one step【231741701474535†L115-L125】.
Understanding this technique and its trade‑offs helps inference
engineers choose the right draft model size, acceptance threshold and
batching strategy for latency‑sensitive applications like chatbots and
code completion tools.

## Features

* **Configurable acceptance rate** – Specify the probability that a
  draft token is accepted by the target (α).
* **Draft and target speeds** – Configure per‑token latencies for
  both models to model differences in compute cost.
* **Draft batch size (γ)** – Control how many tokens the draft model
  proposes each cycle【231741701474535†L150-L151】.
* **Randomised acceptance simulation** – Each token is accepted with
  probability α until the first rejection; the target then
  recomputes the next token.
* **Baseline comparison** – Compute the time taken by the target
  model alone and report the speculative speedup.
* **Command‑line interface** – Run simulations with different
  parameters and output results as JSON.
* **Test suite** – Unit tests ensure that extreme acceptance rates
  behave as expected (speedup > 1 when α=1 and < 1 when α=0).

## Project structure

```text
06-speculative-decoding-simulator/
├── speculative.py        # Simulation logic and CLI
├── tests/
│   └── test_speculative.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run a speculative decoding simulation for 50 tokens with acceptance
rate 0.6, draft speed 0.1 seconds per token, target speed 0.2
seconds per token and draft batch size 4:

```bash
python speculative.py --tokens 50 --accept-rate 0.6 \
    --draft-speed 0.1 --target-speed 0.2 --draft-batch-size 4 --seed 42
```

Sample output:

```json
{
  "speculative_time": 13.15,
  "baseline_time": 10.0,
  "speedup": 0.76
}
```

By varying the acceptance rate you can observe the transition from
speedup to slowdown.  For example, setting `--accept-rate 1.0` yields
a speedup greater than 1, while `--accept-rate 0.0` leads to a
slowdown, as verified in the tests.

## Running the tests

Run the unit tests with `pytest`:

```bash
pytest -q
```

The tests verify that the simulator produces a speedup for full
acceptance (α=1) and a slowdown when no tokens are accepted (α=0).

## Resume bullet

Implemented a speculative decoding simulator modelling draft/token
acceptance and measuring end‑to‑end speedup versus baseline decoding.
The tool allows configuring acceptance rate, draft batch size and
per‑token latencies, offers a CLI for experimentation and includes
unit tests demonstrating the performance extremes of the technique.