#!/usr/bin/env python3
"""
Build Markdown tables from full-run JSON results, with optional filters,
and include the Guess-and-Learn cumulative error **count** (mean ± s.e.).
"""
import argparse, glob, json, statistics as st, sys
from collections import defaultdict
from typing import List, Tuple, Optional

PREFERRED_KEYS = [
    "final_error_rate",   # repo default
    "accuracy", "acc", "score", "metric",
]

def pick_metric_key(d, preferred=None):
    if preferred:
        if preferred in d:
            return preferred
        raise KeyError(f"Metric key '{preferred}' not found. Available top-level keys include: {list(d.keys())[:12]}")
    for k in PREFERRED_KEYS:
        if k in d:
            return k
    # Fallback: derive rate from count if subset is present
    if "final_error_count" in d and isinstance(d.get("params", {}), dict):
        sub = d["params"].get("subset")
        if sub:
            return ("_derived_error_rate", float(d["final_error_count"]) / float(sub))
    raise KeyError("No suitable metric key found. Expected one of "
                   f"{PREFERRED_KEYS} or final_error_count+params.subset.")

def matches(val: str, allow: List[str]) -> bool:
    return (not allow) or (val in allow)

def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def fmt_rate(x: float) -> str:
    return f"{x:.4f}"

def fmt_count(x: float) -> str:
    # counts are integers per seed; mean/SE can be fractional
    return f"{x:.1f}"

def mean_se(values: List[float]) -> Tuple[float, float, int]:
    n = len(values)
    mu = st.mean(values)
    se = (st.pstdev(values) / (n ** 0.5)) if n > 1 else 0.0
    return mu, se, n

def sort_key(k: Tuple[str, str, str, str, Optional[int]]):
    ds, mdl, trk, stg, subset = k
    # Place None budgets last; otherwise numeric ascending
    budget_key = (10**9 if subset is None else int(subset))
    return (ds, mdl, trk, stg, budget_key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob pattern for result JSON files")
    ap.add_argument("--metric-key", default=None, help="JSON key to read for the *rate* column (default: auto; prefers 'final_error_rate')")
    ap.add_argument("--invert", action="store_true", help="Report 1 - rate (e.g., accuracy from error_rate)")
    ap.add_argument("--include-header", action="store_true", help="Print a header row")
    ap.add_argument("--datasets", type=parse_csv_list, default=[], help="Filter datasets (comma-separated)")
    ap.add_argument("--models", type=parse_csv_list, default=[], help="Filter models (comma-separated)")
    ap.add_argument("--tracks", type=parse_csv_list, default=[], help="Filter tracks (comma-separated)")
    ap.add_argument("--strategies", type=parse_csv_list, default=[], help="Filter strategies (comma-separated)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print("No files matched the provided pattern.", file=sys.stderr); sys.exit(1)

    # (ds, mdl, trk, stg, subset) -> lists
    rate_groups   = defaultdict(list)
    count_groups  = defaultdict(list)
    metric_name_any = None
    have_subset_any = False

    for fp in files:
        try:
            with open(fp, "r") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Skip (cannot read JSON): {fp} ({e})", file=sys.stderr); continue

        params = data.get("params", {})
        if not isinstance(params, dict):
            print(f"Skip (missing 'params' block): {fp}", file=sys.stderr); continue

        ds  = params.get("dataset", "?")
        mdl = params.get("model", "?")
        trk = params.get("track", "?")
        stg = params.get("strategy", "?")
        subset = params.get("subset", None)

        if subset is not None:
            have_subset_any = True

        if not (matches(ds, args.datasets) and matches(mdl, args.models) and
                matches(trk, args.tracks) and matches(stg, args.strategies)):
            continue

        # Pick & compute the rate metric
        try:
            picked = pick_metric_key(data, args.metric_key)
            if isinstance(picked, tuple) and picked[0] == "_derived_error_rate":
                rate_val = float(picked[1])
                metric_name = "final_error_rate*"
            else:
                metric_name = picked
                rate_val = float(data[picked])
        except KeyError as e:
            print(f"Skip (no rate metric): {fp} ({e})", file=sys.stderr); continue

        # Optional invert to show accuracy
        if args.invert:
            rate_val = 1.0 - rate_val

        # G&L count (required)
        if "final_error_count" not in data:
            print(f"Skip (missing 'final_error_count'): {fp}", file=sys.stderr); continue
        count_val = float(data["final_error_count"])

        key = (ds, mdl, trk, stg, subset)
        rate_groups[key].append(rate_val)
        count_groups[key].append(count_val)
        metric_name_any = metric_name

    if not rate_groups:
        print("No rows matched filters.", file=sys.stderr); sys.exit(2)

    # Header
    if args.include_header:
        rate_unit  = "(accuracy)" if args.invert else f"( {metric_name_any} )"
        print(f"| Dataset | Model | Track | Strategy | Pool | Mean ± s.e. {rate_unit} | G&L count (mean ± s.e.) | Seeds |")
        print("|:-------:|:-----:|:-----:|:--------:|:------:|:--------------------:|:-------------------------:|:-----:|")

    # Rows (sorted with a key that handles None vs int)
    for key in sorted(rate_groups.keys(), key=sort_key):
        ds, mdl, trk, stg, subset = key
        rates = rate_groups[key]
        counts = count_groups[key]
        r_mu, r_se, n = mean_se(rates)
        c_mu, c_se, _ = mean_se(counts)
        budget = "full" if subset is None else subset
        print(f"| {ds} | {mdl} | {trk} | {stg} | {budget} | {fmt_rate(r_mu)} ± {fmt_rate(r_se)} | {fmt_count(c_mu)} ± {fmt_count(c_se)} | {n} |")

if __name__ == "__main__":
    main()
