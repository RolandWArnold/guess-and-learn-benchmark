#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import math
import re

ACQ_NAMES = {
    "random": "random",
    "confidence": "confidence",
    "least_confidence": "least_confidence",
    "least-confidence": "least_confidence",
    "margin": "margin",
    "entropy": "entropy",
}

def norm_track(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # Accept "po", "pb", "g&l-po", "g&l-pb", etc.
    if "po" in s and "pb" not in s:
        return "PO"
    if "pb" in s:
        return "PB"
    if "so" in s and "sb" not in s:
        return "SO"
    if "sb" in s:
        return "SB"
    return s.upper()

def norm_strategy(s: str, fallback_from_name:str="") -> str:
    cand = (s or "").strip().lower().replace(" ", "_").replace("-", "_")
    if cand in ACQ_NAMES.values():
        return cand
    # Try to derive from filename token if needed
    for k in ACQ_NAMES:
        if k in fallback_from_name.lower():
            return ACQ_NAMES[k]
    return cand or "unknown"

def looks_like_vit_b16(model_name: str, fallback_from_name: str) -> bool:
    # accept variants: "ViT-B/16", "vit-b-16", "vit_b_16", etc.
    s = (model_name or "").lower() + " " + (fallback_from_name or "").lower()
    return ("vit" in s and ("b/16" in s or "b-16" in s or "b_16" in s or "b16" in s))

def looks_like_mnist(dataset: str, fallback_from_name: str) -> bool:
    s = (dataset or "").lower() + " " + (fallback_from_name or "").lower()
    return "mnist" in s and "fashion" not in s

def cumulative_errors_from_json(obj: dict, n: int) -> int:
    # Prefer explicit is_error 0/1 array
    is_err = obj.get("is_error", None)
    if isinstance(is_err, list) and len(is_err) > 0:
        k = min(n, len(is_err))
        # Support ints or booleans inside list
        return int(sum(1 if bool(x) else 0 for x in is_err[:k]))
    # Fallback: try error_history as cumulative
    eh = obj.get("error_history", None)
    if isinstance(eh, list) and len(eh) > 0:
        idx = min(n, len(eh)) - 1
        try:
            return int(eh[idx])
        except Exception:
            pass
    # Last resort: final_error_count if length matches n (rare)
    final_count = obj.get("final_error_count", None)
    if isinstance(final_count, int) and final_count >= 0:
        return int(final_count)
    raise ValueError("Could not derive cumulative error: missing 'is_error' and usable 'error_history'.")

def mean_se(xs):
    if not xs:
        return (float("nan"), float("nan"))
    m = sum(xs)/len(xs)
    if len(xs) == 1:
        return (m, float("nan"))
    var = sum((x-m)**2 for x in xs) / (len(xs)-1)
    se = math.sqrt(var/len(xs))
    return (m, se)

def parse_args():
    ap = argparse.ArgumentParser(
        description="Verify MNIST ViT-B/16 E_n claim (e.g., '>= 229 errors by n=300, mean across seeds')."
    )
    ap.add_argument("results_dir", help="Directory to search recursively for *_results.json files")
    ap.add_argument("--dataset", default="mnist", help="Dataset filter (default: mnist)")
    ap.add_argument("--model", default="vit-b-16", help="Model filter (default: vit-b-16)")
    ap.add_argument("--n", type=int, default=300, help="n for cumulative errors (default: 300)")
    ap.add_argument("--tracks", default="PO,PB", help="Comma list of tracks to include (e.g., 'PO,PB')")
    ap.add_argument("--strategies", default="all", help="Comma list (e.g., 'random,confidence') or 'all'")
    ap.add_argument("--threshold", type=int, default=229, help="PASS if mean >= threshold (default: 229)")
    ap.add_argument("--aggregate", choices=["per_strategy","overall_mean","overall_max"], default="per_strategy",
                    help="How to judge the claim: per_strategy (default), or aggregate over strategies (mean/max).")
    return ap.parse_args()

def main():
    args = parse_args()
    root = Path(args.results_dir)
    if not root.exists():
        print(f"ERROR: path does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    want_tracks = set(t.strip().upper() for t in args.tracks.split(","))
    want_strats = None if args.strategies.lower()=="all" else set(s.strip().lower() for s in args.strategies.split(","))

    rows = []  # (track, strategy, seed, En, file)
    for jf in root.rglob("*.json"):
        name = jf.name
        try:
            obj = json.loads(jf.read_text())
        except Exception:
            continue

        params = obj.get("params", {})
        ds = (params.get("dataset") or "").strip()
        model = (params.get("model") or "").strip()
        track = norm_track(params.get("track") or name)
        strat = norm_strategy(params.get("strategy"), fallback_from_name=name)
        seed = params.get("seed", None)
        if seed is None:
            # try to extract from filename
            m = re.search(r"seed(\d+)", name)
            if m: seed = int(m.group(1))

        # Filters
        if not looks_like_mnist(ds, name): continue
        if not looks_like_vit_b16(model, name): continue
        if want_tracks and track not in want_tracks: continue
        if want_strats and strat not in want_strats: continue

        try:
            En = cumulative_errors_from_json(obj, args.n)
        except Exception:
            continue

        rows.append((track, strat, seed, En, str(jf)))

    if not rows:
        print("No matching files found for MNIST + ViT-B/16 with given filters.", file=sys.stderr)
        sys.exit(1)

    # Group by track/strategy
    by_ts = defaultdict(list)
    for (track, strat, seed, En, fp) in rows:
        by_ts[(track, strat)].append((seed, En, fp))

    print(f"\nFound {len(rows)} matching runs for MNIST + ViT-B/16, n={args.n}.")
    print(f"Tracks included: {', '.join(sorted(set(k[0] for k in by_ts.keys())))}")
    print(f"Strategies included: {', '.join(sorted(set(k[1] for k in by_ts.keys())))}\n")

    header = f"{'Track':<4}  {'Strategy':<17}  {'Seeds':<5}  {'Mean E_n':>8}  {'SE':>7}  {'>= %d ?'%args.threshold:>8}"
    print(header)
    print("-"*len(header))

    overall_vals = []
    per_strategy_pass = []
    for (track, strat), triplets in sorted(by_ts.items()):
        ens = [En for (_, En, _) in triplets]
        m, se = mean_se(ens)
        ok = (m >= args.threshold) if not math.isnan(m) else False
        print(f"{track:<4}  {strat:<17}  {len(ens):<5d}  {m:8.2f}  {se:7.2f}  {str(ok):>8}")
        overall_vals.append(m)
        per_strategy_pass.append(ok)

    # Aggregate checks
    if args.aggregate == "overall_mean":
        m_all, se_all = mean_se(overall_vals)
        ok = m_all >= args.threshold
        print(f"\nAggregate over strategies (mean of means): {m_all:.2f} ± {se_all:.2f}  =>  PASS={ok}")
    elif args.aggregate == "overall_max":
        max_all = max(overall_vals) if overall_vals else float("nan")
        ok = max_all >= args.threshold
        print(f"\nAggregate over strategies (max of means): {max_all:.2f}  =>  PASS={ok}")
    else:
        # per_strategy: at least one strategy passes
        ok = any(per_strategy_pass)
        print(f"\nPer-strategy PASS if any strategy's mean >= {args.threshold}: PASS={ok}")

    # Helpful hints if nothing passes
    if not ok:
        print("\nHint: You claimed '>= %d by n=%d (mean across seeds)'. Consider:\n"
              "- Confirm the track (PO vs PB) used for the claim.\n"
              "- Confirm which acquisition strategy the claim referenced.\n"
              "- If your figures average over strategies, re-run with --aggregate overall_mean.\n" % (args.threshold, args.n))

if __name__ == "__main__":
    main()
