#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from statistics import mean, median
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


FEATURE_KEYS = [f"feat{i}" for i in range(1, 30)]  # feat1..feat29

def parse_args():
    ap = argparse.ArgumentParser(
        description="Normalize AR feature time series JSON and plot selected variables."
    )
    ap.add_argument("--input-json", required=True, help="Path to input JSON (from make_arfeat_ts_data.py).")
    ap.add_argument("--output-json", required=True, help="Path to output JSON with normalized features.")
    ap.add_argument("--norm-min", type=float, default=0.0, help="Normalization lower bound (default: 0.0).")
    ap.add_argument("--norm-max", type=float, default=1.0, help="Normalization upper bound (default: 1.0).")
    ap.add_argument("--global-norm", action="store_true",
                help="If set, use global per-feature min/max across all series.")
    ap.add_argument(
        "--variables-to-be-plotted",
        type=str,
        default="",
        help='Comma-separated indices to plot, e.g. "0,3,7,8,20" (0-based) or "1,4,8" (1-based). '
             "You may also pass names like feat1,feat4.",
    )
    ap.add_argument(
        "--series-index",
        type=int,
        default=0,
        help="Which series in the JSON 'data' list to plot (default: 0).",
    )
    ap.add_argument(
        "--save-plot",
        type=str,
        default="",
        help="Optional path to save the plot (PNG/PDF). If empty, the plot is shown interactively.",
    )
    
    ap.add_argument("--scaler", choices=["minmax", "zscore", "robust"], default="minmax", help="Type of global scaling per feature when --global-norm is set. "
        "'minmax' uses global min/max, 'zscore' uses mean/std, "
        "'robust' uses median/MAD."
    )
    ap.add_argument("--save-stats", type=str, default=None, help="Path to save fitted global stats as JSON (train time).")
    ap.add_argument("--load-stats", type=str, default=None, help="Path to load precomputed stats JSON (inference time).")

    return ap.parse_args()




def parse_var_list(s: str):
    """Accept 0-based indices, 1-based indices, or names like 'feat7'."""
    if not s:
        return []
    out = set()
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        if t.lower().startswith("feat"):
            try:
                k = int(t[4:])
                if 1 <= k <= 29:
                    out.add(f"feat{k}")
            except Exception:
                pass
        else:
            # numeric: try 0-based first, then 1-based
            try:
                idx = int(t)
                if 0 <= idx <= 28:      # 0-based
                    out.add(f"feat{idx+1}")
                elif 1 <= idx <= 29:    # 1-based
                    out.add(f"feat{idx}")
            except Exception:
                pass
    return [k for k in FEATURE_KEYS if k in out]

def gen_timestamps(date_start_str: str, npoints: int, dt_minutes: int):
    t0 = datetime.strptime(date_start_str, "%Y-%m-%d %H:%M:%S")
    return [t0 + timedelta(minutes=dt_minutes * k) for k in range(npoints)]

def minmax_normalize(seq, a=0.0, b=1.0):
    """Normalize a list of floats to [a, b]. If constant series, put all at midpoint."""
    if not seq:
        return []
    lo = min(seq)
    hi = max(seq)
    if hi == lo:
        mid = (a + b) / 2.0
        return [mid] * len(seq)
    scale = (b - a) / (hi - lo)
    return [a + (x - lo) * scale for x in seq]

def normalize_record(rec, a=0.0, b=1.0):
    """Return a *new* record with normalized features; metadata unchanged."""
    out = dict(rec)  # shallow copy
    for k in FEATURE_KEYS:
        if k in rec and isinstance(rec[k], list):
            out[k] = minmax_normalize(rec[k], a=a, b=b)
    return out


def _mad(vals, med):
    # median absolute deviation (normalized to be comparable to std for normal data)
    abs_dev = [abs(v - med) for v in vals]
    m = median(abs_dev)
    return 1.4826 * m  # consistency factor

def compute_global_stats(records, feature_keys, scaler):
    stats = {}
    for k in feature_keys:
        # flatten across all series
        allv = []
        for rec in records:
            seq = rec.get(k, [])
            if isinstance(seq, list):
                allv.extend([x for x in seq if isinstance(x, (int, float)) and not math.isnan(x)])
        if not allv:
            stats[k] = {}
            continue
        if scaler == "minmax":
            stats[k] = {"min": float(min(allv)), "max": float(max(allv))}
        elif scaler == "zscore":
            mu = float(mean(allv))
            # population std
            var = float(mean([(x - mu) ** 2 for x in allv]))
            sigma = float(math.sqrt(var))
            stats[k] = {"mean": mu, "std": sigma if sigma > 0 else 1.0}
        elif scaler == "robust":
            med = float(median(allv))
            mad = float(_mad(allv, med))
            stats[k] = {"median": med, "mad": mad if mad > 0 else 1.0}
    return stats

def apply_global_scaling(rec, feature_keys, stats, scaler, a=0.0, b=1.0):
    out = dict(rec)
    for k in feature_keys:
        seq = rec.get(k, None)
        if not isinstance(seq, list):
            continue
        s = stats.get(k, {})
        if scaler == "minmax":
            lo, hi = s.get("min"), s.get("max")
            if lo is None or hi is None or hi == lo:
                mid = (a + b) / 2.0
                out[k] = [mid] * len(seq)
            else:
                scale = (b - a) / (hi - lo)
                out[k] = [a + (x - lo) * scale for x in seq]
        elif scaler == "zscore":
            mu, sd = s.get("mean"), s.get("std", 1.0)
            if mu is None:
                out[k] = seq
            else:
                sd = sd if sd > 0 else 1.0
                out[k] = [(x - mu) / sd for x in seq]
        elif scaler == "robust":
            med, mad = s.get("median"), s.get("mad", 1.0)
            if med is None:
                out[k] = seq
            else:
                mad = mad if mad > 0 else 1.0
                out[k] = [(x - med) / mad for x in seq]
    return out

def main():
    args = parse_args()
    assert args.norm_max > args.norm_min, "--norm-max must be > --norm-min"

    with open(args.input_json, "r") as f:
        payload = json.load(f)

    if "data" not in payload or not isinstance(payload["data"], list):
        raise ValueError("Input JSON must contain a top-level 'data' list.")

    # after loading payload
    global_min = {k: float("inf") for k in FEATURE_KEYS}
    global_max = {k: float("-inf") for k in FEATURE_KEYS}
    if args.global_norm:
        for rec in payload["data"]:
            for k in FEATURE_KEYS:
                if k in rec and isinstance(rec[k], list) and rec[k]:
                    vmin = min(rec[k])
                    vmax = max(rec[k])
                    if vmin < global_min[k]: global_min[k] = vmin
                    if vmax > global_max[k]: global_max[k] = vmax

    print("global_min")
    print(global_min)
    print("global_max")
    print(global_max)
    
    
    if args.global_norm:
        # Normalize global per-feature
        if args.load_stats:
            with open(args.load_stats, "r") as fh:
                stats = json.load(fh)
        else:
            stats = compute_global_stats(payload["data"], FEATURE_KEYS, args.scaler)
            if args.save_stats:
                with open(args.save_stats, "w") as fh:
                    json.dump(stats, fh, indent=2)

        norm_data = [apply_global_scaling(rec, FEATURE_KEYS, stats, args.scaler, a=args.norm_min, b=args.norm_max) for rec in payload["data"]]
        
    else:
        # Normalize every series independently (per-series min–max for each feature)
        norm_data = [
            normalize_record(rec, a=args.norm_min, b=args.norm_max)
            for rec in payload["data"]
        ]

    payload["data"] = norm_data
    
    
    # Normalize every series independently (per-series min–max for each feature)
    #norm_data = []
    #for rec in payload["data"]:
    #    norm_data.append(normalize_record(rec, a=args.norm_min, b=args.norm_max))

    # Write normalized JSON with the same schema
    out_payload = {"data": norm_data}
    with open(args.output_json, "w") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    # Plot (optional)
    vars_to_plot = parse_var_list(args.variables_to_be_plotted)
    if vars_to_plot:
        idx = max(0, min(args.series_index, len(norm_data) - 1))
        rec = norm_data[idx]
        npoints = int(rec["npoints"])
        dt_minutes = int(rec["dt"])
        ts = gen_timestamps(rec["date_start"], npoints, dt_minutes)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for k in vars_to_plot:
            if k in rec:
                ax.plot(ts, rec[k], label=k)

        ax.set_title(f"AR {rec.get('ar', '?')}, label={rec.get('flare_type', '?')}")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Normalized feature value")
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.grid(True, alpha=0.3)
        if vars_to_plot:
            ax.legend(loc="best", ncols=2)

        plt.tight_layout()
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        else:
            plt.show()

    print(f"Normalized {len(norm_data)} series → {args.output_json}")
    if vars_to_plot:
        print(f"Plotted {len(vars_to_plot)} variables for series index {args.series_index}.")

if __name__ == "__main__":
    main()

