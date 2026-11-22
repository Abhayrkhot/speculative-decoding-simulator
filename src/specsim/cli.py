"""Command-line interface: ``specsim <command> [options]``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .analytic import expected_speedup, optimal_gamma
from .latency import HARDWARE, MODELS, ConstantLatency, LatencyModel, RooflineLatency
from .simulate import simulate


def _frange(spec: str) -> list[float]:
    """Parse ``"0.8"``, ``"0.5,0.7,0.9"`` or an inclusive range ``"0.5:0.95:0.05"``."""
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        if step <= 0:
            raise argparse.ArgumentTypeError("range step must be positive")
        n = round((stop - start) / step) + 1
        return [round(start + i * step, 10) for i in range(max(n, 0))]
    return [float(x) for x in spec.split(",") if x]


def _irange(spec: str) -> list[int]:
    """Parse ``"4"``, ``"1,2,4"`` or an inclusive range ``"1:8"``."""
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        step = parts[2] if len(parts) == 3 else 1
        return list(range(parts[0], parts[1] + 1, step))
    return [int(x) for x in spec.split(",") if x]


def _add_latency_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("latency model (pick one; default --c 0.05)")
    g.add_argument("--c", type=float, help="draft/target cost ratio (constant-latency model)")
    g.add_argument("--target-ms", type=float, help="target forward pass, ms")
    g.add_argument("--draft-ms", type=float, help="draft forward pass, ms")
    g.add_argument(
        "--roofline",
        nargs=2,
        metavar=("TARGET", "DRAFT"),
        help=f"roofline model from presets: {', '.join(MODELS)}",
    )
    g.add_argument("--gpu", default="h100-sxm", choices=sorted(HARDWARE))
    g.add_argument("--context", type=int, default=1024, help="KV-cache length (roofline)")
    g.add_argument("--overhead-ms", type=float, default=0.0, help="fixed per-forward overhead")
    p.add_argument("--batch", type=int, default=1, help="sequences decoded together")


def _latency(args: argparse.Namespace) -> LatencyModel:
    if args.roofline:
        t, d = args.roofline
        for name in (t, d):
            if name not in MODELS:
                raise SystemExit(f"unknown model preset {name!r}; choose from {', '.join(MODELS)}")
        return RooflineLatency(
            MODELS[t],
            MODELS[d],
            HARDWARE[args.gpu],
            context=args.context,
            target_overhead_s=args.overhead_ms / 1e3,
            draft_overhead_s=args.overhead_ms / 1e3,
        )
    if args.target_ms is not None or args.draft_ms is not None:
        if args.target_ms is None or args.draft_ms is None:
            raise SystemExit("--target-ms and --draft-ms must be given together")
        o = args.overhead_ms
        return ConstantLatency((args.target_ms + o) / 1e3, (args.draft_ms + o) / 1e3)
    if args.overhead_ms:
        raise SystemExit("--overhead-ms needs --target-ms/--draft-ms or --roofline")
    return ConstantLatency.from_ratio(args.c if args.c is not None else 0.05)


def _print_table(rows: list[dict[str, Any]], cols: Sequence[str]) -> None:
    def fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    cells = [[fmt(r[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths, strict=True)))
    for row in cells:
        print("  ".join(v.rjust(w) for v, w in zip(row, widths, strict=True)))


def cmd_simulate(args: argparse.Namespace) -> int:
    lat = _latency(args)
    res = simulate(args.alpha, args.gamma, lat, args.tokens, args.trials, args.batch, args.seed)
    out = res.to_dict()
    out["analytic_speedup"] = expected_speedup(args.alpha, args.gamma, lat, args.batch)
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    lo, hi = res.speedup_ci95
    print(f"speedup            {res.speedup:.3f}x  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"analytic speedup   {out['analytic_speedup']:.3f}x  (infinite-horizon closed form)")
    print(f"per-generation     p5 {res.speedup_p05:.3f}x  p95 {res.speedup_p95:.3f}x")
    print(f"tokens per cycle   {res.tokens_per_cycle:.3f}")
    print(f"acceptance rate    {res.acceptance_rate:.3f}  (accepted / verified)")
    print(f"draft efficiency   {res.draft_efficiency:.3f}  (accepted / drafted)")
    print(f"target calls saved {res.target_calls_saved:.1%}")
    return 0


def cmd_optimal_gamma(args: argparse.Namespace) -> int:
    lat = _latency(args)
    rows = []
    for a in _frange(args.alpha):
        g, s = optimal_gamma(a, lat, args.max_gamma, args.batch)
        rows.append({"alpha": a, "best_gamma": g, "speedup": s})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_table(rows, ["alpha", "best_gamma", "speedup"])
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    lat = _latency(args)
    alphas, gammas = _frange(args.alpha), _irange(args.gamma)
    rows: list[dict[str, Any]] = []
    for a in alphas:
        for g in gammas:
            row: dict[str, Any] = {
                "alpha": a,
                "gamma": g,
                "analytic": expected_speedup(a, g, lat, args.batch),
            }
            if args.trials:
                r = simulate(a, g, lat, args.tokens, args.trials, args.batch, args.seed)
                row.update(simulated=r.speedup, ci_low=r.speedup_ci95[0], ci_high=r.speedup_ci95[1])
            rows.append(row)
    cols = list(rows[0])
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}", file=sys.stderr)
    if args.plot:
        from .plots import speedup_heatmap

        speedup_heatmap(alphas, gammas, lat, args.plot, batch=args.batch)
        print(f"wrote {args.plot}", file=sys.stderr)
    if not args.csv and not args.plot:
        _print_table(rows, cols)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import calibration_report

    rep = calibration_report(args.results)
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    meta = rep["meta"]
    print(f"{meta['draft']} -> {meta['target']} on {meta['device']}")
    lat = rep["latency"]
    print(
        f"latency (ctx {lat['context']}): target {lat['target_fixed_ms']:.2f} ms "
        f"+ {lat['target_per_token_ms']:.3f} ms/token, draft {lat['draft_fixed_ms']:.2f} ms "
        f"(c = {lat['c']:.3f})"
    )
    rows = [
        {
            "task": r["task"],
            "T": r["temperature"],
            "alpha": r["alpha"],
            "alpha_pos1": r["positional"][0],
            "alpha_pos4": r["positional"][min(3, len(r["positional"]) - 1)],
            "cycles": r["cycles"],
        }
        for r in rep["acceptance"]
    ]
    _print_table(rows, ["task", "T", "alpha", "alpha_pos1", "alpha_pos4", "cycles"])
    gm = rep["greedy_match"]
    if gm["runs"]:
        print(
            f"greedy outputs identical to target-only: {gm['exact_match_rate']:.0%} of "
            f"{gm['runs']} runs (mean matching prefix {gm['mean_matching_prefix']:.0%})"
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .calibrate import validate

    rep = validate(args.results, acceptance=args.acceptance)
    if args.plot:
        from .plots import validation_plot

        validation_plot(rep, args.plot)
        print(f"wrote {args.plot}", file=sys.stderr)
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    _print_table(
        rep["rows"],
        ["task", "temperature", "gamma", "measured", "predicted", "error_pct", "predicted_raw"],
    )
    s = rep["summary"]
    print(
        f"\nmean |error| {s['mae_pct']:.1f}% (max {s['max_abs_error_pct']:.1f}%); "
        f"without per-cycle overhead {s['mae_raw_pct']:.1f}% (max "
        f"{s['max_abs_error_raw_pct']:.1f}%); best measured {s['best_measured']:.2f}x"
    )
    return 0


def cmd_lossless(args: argparse.Namespace) -> int:
    from .sampling import lossless_check

    rep = lossless_check(
        vocab=args.vocab,
        gamma=args.gamma,
        divergence=args.divergence,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps(rep, indent=2))
    return 0 if rep["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="specsim",
        description="Simulate, calibrate and validate speculative decoding speedups.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="Monte Carlo simulation of one configuration")
    s.add_argument("--alpha", type=float, default=0.8, help="acceptance rate")
    s.add_argument("--gamma", type=int, default=4, help="draft tokens per cycle")
    s.add_argument("--tokens", type=int, default=256, help="tokens per generation")
    s.add_argument("--trials", type=int, default=2000, help="generations to simulate")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--json", action="store_true")
    _add_latency_args(s)
    s.set_defaults(func=cmd_simulate)

    o = sub.add_parser("optimal-gamma", help="best draft length for given acceptance rate(s)")
    o.add_argument("--alpha", default="0.5:0.95:0.05", help="value, list or start:stop:step")
    o.add_argument("--max-gamma", type=int, default=16)
    o.add_argument("--json", action="store_true")
    _add_latency_args(o)
    o.set_defaults(func=cmd_optimal_gamma)

    w = sub.add_parser("sweep", help="speedup over a grid of alpha x gamma")
    w.add_argument("--alpha", default="0.5:0.95:0.05")
    w.add_argument("--gamma", default="1:8")
    w.add_argument("--trials", type=int, default=0, help="also run Monte Carlo (0 = analytic)")
    w.add_argument("--tokens", type=int, default=256)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--csv", help="write rows to CSV")
    w.add_argument("--plot", help="write a heatmap PNG (needs matplotlib)")
    _add_latency_args(w)
    w.set_defaults(func=cmd_sweep)

    c = sub.add_parser("calibrate", help="fit latency and acceptance from a benchmark JSON")
    c.add_argument("results", help="output of bench/spec_bench.py")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("validate", help="compare predicted vs measured speedup")
    v.add_argument("results", help="output of bench/spec_bench.py")
    v.add_argument(
        "--acceptance",
        choices=["positional", "iid"],
        default="positional",
        help="acceptance model fitted from traces of the *other* gammas",
    )
    v.add_argument("--plot", help="write predicted-vs-measured PNG")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_validate)

    ll = sub.add_parser("lossless-check", help="verify rejection sampling preserves the target")
    ll.add_argument("--vocab", type=int, default=32)
    ll.add_argument("--gamma", type=int, default=4)
    ll.add_argument("--divergence", type=float, default=1.0, help="draft/target logit noise")
    ll.add_argument("--samples", type=int, default=200_000)
    ll.add_argument("--seed", type=int, default=0)
    ll.set_defaults(func=cmd_lossless)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as e:
        print(f"specsim: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
