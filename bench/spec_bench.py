"""Measure real speculative decoding on a GPU.

Runs a draft/target model pair through a hand-written speculative decoding
loop (with KV-cache rollback) and records everything the simulator needs to be
calibrated and validated against reality:

* per-cycle acceptance traces ``[gamma_used, n_accepted]`` per prompt, task,
  gamma and temperature;
* a latency profile: wall time of one forward pass that appends ``k`` tokens to
  a KV cache of a given context length, for both models;
* end-to-end wall-clock time of speculative decoding versus plain
  autoregressive decoding with the target model (same code path, gamma=0);
* a greedy losslessness check (speculative output vs. target-only output);
* optionally, Hugging Face's built-in assisted generation as an external
  reference point.

The output JSON is consumed by ``specsim calibrate`` / ``specsim validate``.

Example::

    python bench/spec_bench.py --draft Qwen/Qwen2.5-0.5B-Instruct \
        --target Qwen/Qwen2.5-7B-Instruct --out results/qwen2.5-0.5b-7b.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# Prompts: four task families with very different draft/target agreement.
# --------------------------------------------------------------------------

_PASSAGE_A = (
    "The city council voted on Tuesday to convert the old rail yard on the east "
    "side into a mixed-use district. The plan keeps two of the original brick "
    "warehouses, which will become a public market and a maker space, and adds "
    "roughly 900 apartments, a third of them priced below market rate. Opponents "
    "argued that the area lacks transit and that traffic on Harbor Avenue is "
    "already at capacity; supporters pointed to a new bus rapid transit line "
    "scheduled to open in two years. Construction is expected to start next "
    "spring and finish in phases over six years."
)
_PASSAGE_B = (
    "Our quarterly review found that average page load time rose from 1.8 to 2.6 "
    "seconds after the image service migration. Most of the regression comes from "
    "thumbnails being generated on request instead of at upload time, and from a "
    "missing cache header on the CDN. Reverting thumbnail generation to the upload "
    "pipeline and restoring the header should bring load time back under two "
    "seconds. The team also recommends adding a load-time budget to the release "
    "checklist so similar regressions are caught before launch."
)

PROMPTS: dict[str, list[str]] = {
    "code": [
        "Write a Python function that returns the n-th Fibonacci number using "
        "memoization, with a docstring and type hints.",
        "Write a Python class implementing an LRU cache with get and put methods "
        "in O(1) time. Include a short usage example.",
        "Write a function in Python that parses a CSV file and returns the average "
        "of a numeric column given its name. Handle missing values.",
        "Implement binary search in Python over a sorted list and return the index "
        "of the target or -1. Add unit tests using pytest.",
        "Write a JavaScript function that debounces another function, then show "
        "how to use it on a window resize event.",
    ],
    "math": [
        "A train travels 180 km at 60 km/h and then 120 km at 40 km/h. What is its "
        "average speed for the whole trip? Show your reasoning step by step.",
        "Solve for x: 3(x - 4) + 2 = 5x - 14. Explain each step.",
        "A rectangle has a perimeter of 54 cm and its length is twice its width. "
        "Find its area. Show all steps.",
        "If a shirt costs $40 after a 20% discount, what was the original price? "
        "Explain the calculation.",
        "What is the sum of all integers from 1 to 200 that are divisible by 3 or "
        "5? Show the method.",
    ],
    "chat": [
        "What are some good habits for staying focused while working from home?",
        "Explain how a refrigerator keeps food cold, in simple terms.",
        "I'm planning a weekend trip to the mountains in autumn. What should I pack?",
        "What's the difference between weather and climate?",
        "Give me some tips for writing a clear and friendly email to a new colleague.",
    ],
    "summarize": [
        f"Summarize the following article in three sentences:\n\n{_PASSAGE_A}",
        f"List the key facts from this article as bullet points:\n\n{_PASSAGE_A}",
        f"Summarize this report for an executive in two sentences:\n\n{_PASSAGE_B}",
        f"Rewrite this report as a short status update email:\n\n{_PASSAGE_B}",
        f"Extract every number mentioned in this text and say what it refers to:\n\n{_PASSAGE_A}",
    ],
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load(name: str, device: str, dtype: torch.dtype) -> Any:
    kwargs: dict[str, Any] = {"device_map": device, "attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)
    model.eval()
    return model


def encode(tok: Any, prompt: str, device: str) -> torch.Tensor:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tok(text, return_tensors="pt").input_ids.to(device)


def eos_ids(tok: Any) -> set[int]:
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return ids


# --------------------------------------------------------------------------
# Speculative decoding with KV-cache rollback
# --------------------------------------------------------------------------


@torch.inference_mode()
def speculative_generate(
    target: Any,
    draft: Any | None,
    prompt_ids: torch.Tensor,
    gamma: int,
    max_new: int,
    temperature: float,
    vocab: int,
    stop: set[int],
    gen: torch.Generator,
) -> dict[str, Any]:
    """Generate up to ``max_new`` tokens. ``gamma == 0`` is plain target decoding.

    Invariant between cycles: both KV caches hold ``seq[:-1]`` (or a prefix of
    it for the draft); the last token of ``seq`` is pending.

    For temperature > 0 this is the lossless rejection-sampling scheme of
    Leviathan et al. (2023) / Chen et al. (2023): accept draft token x with
    probability min(1, p(x)/q(x)); on rejection, resample from
    norm(max(0, p - q)); if all gamma drafts are accepted, sample a bonus token
    from p. For temperature == 0 both models are greedy, so a draft token is
    accepted iff it equals the target's argmax.
    """
    dev = prompt_ids.device
    seq: list[int] = prompt_ids[0].tolist()
    n_prompt = len(seq)

    sync()
    t_start = time.perf_counter()
    t_cache = target(prompt_ids[:, :-1], use_cache=True).past_key_values
    d_cache = None
    if gamma > 0 and draft is not None:
        d_cache = draft(prompt_ids[:, :-1], use_cache=True).past_key_values
    sync()
    t_prefill = time.perf_counter() - t_start

    cycles: list[list[int]] = []
    while True:
        produced = len(seq) - n_prompt
        if produced >= max_new or (produced > 0 and seq[-1] in stop):
            break
        k = min(gamma, max_new - produced - 1)

        # ---- draft phase: k sequential single-token steps ----
        drafts: list[int] = []
        qs: list[torch.Tensor] = []
        if k > 0:
            assert draft is not None and d_cache is not None
            pending = seq[d_cache.get_seq_length() :]
            for _ in range(k):
                logits = draft(
                    torch.tensor([pending], device=dev), past_key_values=d_cache, use_cache=True
                ).logits[0, -1, :vocab]
                if temperature == 0:
                    x = int(torch.argmax(logits))
                else:
                    q = torch.softmax(logits.float() / temperature, dim=-1)
                    x = int(torch.multinomial(q, 1, generator=gen))
                    qs.append(q)
                drafts.append(x)
                pending = [x]

        # ---- verify phase: ONE target pass over [last token] + drafts ----
        t_in = seq[t_cache.get_seq_length() :] + drafts
        logits = target(
            torch.tensor([t_in], device=dev), past_key_values=t_cache, use_cache=True
        ).logits[0, -(k + 1) :, :vocab]

        n_acc = 0
        new_tok: int | None = None
        if temperature == 0:
            best = torch.argmax(logits, dim=-1).tolist()
            for i, x in enumerate(drafts):
                if x != best[i]:
                    new_tok = best[i]
                    break
                n_acc += 1
            if new_tok is None:
                new_tok = best[k]
        else:
            p_all = torch.softmax(logits.float() / temperature, dim=-1)
            for i, x in enumerate(drafts):
                r = torch.rand((), device=dev, generator=gen)
                if bool(r * qs[i][x] <= p_all[i, x]):
                    n_acc += 1
                    continue
                resid = torch.clamp(p_all[i] - qs[i], min=0.0)
                total = resid.sum()
                dist = resid / total if bool(total > 0) else p_all[i]
                new_tok = int(torch.multinomial(dist, 1, generator=gen))
                break
            if new_tok is None:
                new_tok = int(torch.multinomial(p_all[k], 1, generator=gen))

        base = len(seq)
        accepted = drafts[:n_acc]
        # Stop early if an accepted draft token is an end-of-sequence token.
        for j, x in enumerate(accepted):
            if x in stop:
                accepted = accepted[: j + 1]
                new_tok = None
                break
        seq.extend(accepted)
        if new_tok is not None:
            seq.append(new_tok)
        cycles.append([k, n_acc])

        # ---- roll back both caches to the accepted prefix ----
        t_cache.crop(base + len(accepted))
        if d_cache is not None:
            d_cache.crop(base + len(accepted))

    sync()
    t_total = time.perf_counter() - t_start
    new = seq[n_prompt : n_prompt + max_new]
    return {
        "tokens": new,
        "n_new": len(new),
        "wall_s": t_total,
        "prefill_s": t_prefill,
        "decode_s": t_total - t_prefill,
        "cycles": cycles,
    }


# --------------------------------------------------------------------------
# Latency profile
# --------------------------------------------------------------------------


@torch.inference_mode()
def profile_forward(
    model: Any, ctx_len: int, ks: list[int], reps: int, device: str, vocab: int
) -> dict[str, float]:
    """Median wall time (ms) of one forward pass appending k tokens to a cache."""
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(1000, 20000, (1, ctx_len), generator=g).to(device)
    cache = model(ids, use_cache=True).past_key_values
    out: dict[str, float] = {}
    for k in ks:
        x = torch.randint(1000, 20000, (1, k), generator=g).to(device)
        times = []
        for r in range(reps + 3):
            sync()
            t = time.perf_counter()
            logits = model(x, past_key_values=cache, use_cache=True).logits
            _ = int(torch.argmax(logits[0, -1, :vocab]))  # host sync, as in the real loop
            sync()
            if r >= 3:
                times.append(time.perf_counter() - t)
            cache.crop(ctx_len)
        out[str(k)] = statistics.median(times) * 1e3
    return out


# --------------------------------------------------------------------------
# Hugging Face assisted generation (external reference)
# --------------------------------------------------------------------------


@torch.inference_mode()
def hf_reference(
    target: Any, draft: Any, tok: Any, prompts: list[str], gamma: int, max_new: int, device: str
) -> dict[str, Any]:
    res: dict[str, Any] = {"gamma": gamma, "baseline_s": 0.0, "assisted_s": 0.0, "tokens": 0}
    # HF reads drafting knobs from the assistant's generation config. Fix gamma
    # so the comparison with our loop is like-for-like.
    gc = draft.generation_config
    gc.num_assistant_tokens = gamma
    gc.num_assistant_tokens_schedule = "constant"
    if hasattr(gc, "assistant_confidence_threshold"):
        gc.assistant_confidence_threshold = 0.0
    try:
        for p in prompts:
            ids = encode(tok, p, device)
            common = {
                "max_new_tokens": max_new,
                "do_sample": False,
                "pad_token_id": tok.eos_token_id,
            }
            sync()
            t = time.perf_counter()
            out = target.generate(ids, **common)
            sync()
            res["baseline_s"] += time.perf_counter() - t
            res["tokens"] += out.shape[1] - ids.shape[1]
            sync()
            t = time.perf_counter()
            target.generate(ids, assistant_model=draft, **common)
            sync()
            res["assisted_s"] += time.perf_counter() - t
        res["speedup"] = res["baseline_s"] / res["assisted_s"]
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gammas", default="1,2,3,4,5,6,8")
    ap.add_argument("--temperatures", default="0,0.7")
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--prompts-per-task", type=int, default=5)
    ap.add_argument("--profile-ctx", default="256,1024")
    ap.add_argument("--profile-ks", default="1,2,3,4,5,6,7,8,9,12,16,24,32")
    ap.add_argument("--profile-reps", type=int, default=20)
    ap.add_argument("--hf-reference", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    gammas = [int(g) for g in args.gammas.split(",")]
    temps = [float(t) for t in args.temperatures.split(",")]
    tasks = {k: v[: args.prompts_per_task] for k, v in PROMPTS.items()}

    def log(msg: str) -> None:
        print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)

    log(f"loading {args.target} and {args.draft} on {device} ({dtype})")
    tok = AutoTokenizer.from_pretrained(args.target)
    target = load(args.target, device, dtype)
    draft = load(args.draft, device, dtype)
    vocab = min(target.config.vocab_size, draft.config.vocab_size, len(tok))
    stop = eos_ids(tok)

    result: dict[str, Any] = {
        "meta": {
            "draft": args.draft,
            "target": args.target,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
            "dtype": str(dtype),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "date": dt.datetime.now().isoformat(timespec="seconds"),
            "max_new": args.max_new,
            "gammas": gammas,
            "temperatures": temps,
            "prompts_per_task": args.prompts_per_task,
        },
        "latency_ms": {},
        "runs": [],
    }

    # Warm-up (CUDA context, cuBLAS handles, allocator).
    warm = encode(tok, "Hello!", device)
    speculative_generate(
        target, draft, warm, 2, 8, 0.0, vocab, stop, torch.Generator(device=device).manual_seed(0)
    )

    log("profiling forward latency")
    ks = [int(k) for k in args.profile_ks.split(",")]
    for ctx in (int(c) for c in args.profile_ctx.split(",")):
        result["latency_ms"][str(ctx)] = {
            "target": profile_forward(target, ctx, ks, args.profile_reps, device, vocab),
            "draft": profile_forward(draft, ctx, [1, 2, 4], args.profile_reps, device, vocab),
        }
        lat = result["latency_ms"][str(ctx)]
        log(
            f"  ctx={ctx}: target k=1 {lat['target']['1']:.2f} ms, "
            f"draft k=1 {lat['draft']['1']:.2f} ms"
        )

    encoded = {t: [encode(tok, p, device) for p in ps] for t, ps in tasks.items()}
    for temp in temps:
        for gamma in [0, *gammas]:
            t0 = time.perf_counter()
            n_tok = 0
            for task, id_list in encoded.items():
                for pi, ids in enumerate(id_list):
                    seed = args.seed * 1_000_003 + pi * 7919 + gamma * 104_729 + int(temp * 1000)
                    gen = torch.Generator(device=device).manual_seed(seed)
                    r = speculative_generate(
                        target, draft, ids, gamma, args.max_new, temp, vocab, stop, gen
                    )
                    n_tok += r["n_new"]
                    result["runs"].append(
                        {
                            "task": task,
                            "prompt_id": pi,
                            "prompt_tokens": int(ids.shape[1]),
                            "gamma": gamma,
                            "temperature": temp,
                            **r,
                        }
                    )
            log(
                f"T={temp} gamma={gamma}: {n_tok} tokens in {time.perf_counter() - t0:.1f}s "
                f"({n_tok / (time.perf_counter() - t0):.1f} tok/s)"
            )
            # Persist partial results so a timeout never loses everything.
            with open(args.out, "w") as f:
                json.dump(result, f)

    if args.hf_reference:
        log("running Hugging Face assisted-generation reference")
        ref_prompts = [ps[0] for ps in tasks.values()]
        result["hf_reference"] = hf_reference(
            target, draft, tok, ref_prompts, 4, args.max_new, device
        )
        log(f"  {result['hf_reference']}")

    with open(args.out, "w") as f:
        json.dump(result, f)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
