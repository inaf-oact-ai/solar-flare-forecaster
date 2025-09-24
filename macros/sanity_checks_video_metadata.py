#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity checks for HMI video metadata JSON files.

Implements checks:
  (2) Temporal integrity per clip: verify uniform frame spacing; print a sample of clips.
  (3) Cross-split AR uniqueness: an AR must appear in only one split.
  (4) Distribution report per split: class counts (by "label") and clips-per-AR histogram.

Usage:
  python sanity_checks_video_metadata.py \
      --train path/to/train.json \
      --cv path/to/cv.json \
      --test path/to/test.json \
      --cadence-minutes 12 \
      --frame-step-minutes 12 \
      --sample-clips 10 \
      --outdir reports/
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# --- Timestamp parsing helpers ------------------------------------------------

TS_PATTERNS = [
    # 2010-07-18_19-36-00 or 2010-07-18-19-36-00
    (re.compile(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})[_-](\d{2})[-_](\d{2})[-_](\d{2})'), "%Y-%m-%d %H:%M:%S"),
    # 20100718_193600
    (re.compile(r'(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})'), "%Y%m%d %H%M%S"),
    # 2010-07-18T19-36-00 or 2010-07-18T19:36:00
    (re.compile(r'(\d{4})-(\d{2})-(\d{2})[T_](\d{2})[:\-](\d{2})[:\-](\d{2})'), "%Y-%m-%d %H:%M:%S"),
    # 2010-07-18 19:36:00 (spaces)
    (re.compile(r'(\d{4})-(\d{2})-(\d{2})[ _](\d{2}):(\d{2}):(\d{2})'), "%Y-%m-%d %H:%M:%S"),
]

def parse_timestamp_from_path(p: str):
    fname = os.path.basename(p)
    for pat, fmt in TS_PATTERNS:
        m = pat.search(fname)
        if m:
            g = m.groups()
            if fmt == "%Y%m%d %H%M%S":
                s = f"{g[0]}{g[1]}{g[2]} {g[3]}{g[4]}{g[5]}"
            else:
                s = f"{g[0]}-{g[1]}-{g[2]} {g[3]}:{g[4]}:{g[5]}"
            return datetime.strptime(s, fmt)
    raise ValueError(f"Cannot parse timestamp from filename: {fname}")

# --- Checks -------------------------------------------------------------------

def temporal_integrity(sample, expected_step_min: int, verbose: bool = True):
    frames = sample.get("filepaths") or sample.get("frames") or []
    if len(frames) < 2:
        return True, [], []
    ts = [parse_timestamp_from_path(p) for p in frames]
    deltas = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
    expected = timedelta(minutes=expected_step_min)
    bad_idx = [i for i, d in enumerate(deltas) if abs(d - expected) > timedelta(seconds=1)]
    ok = (len(bad_idx) == 0)
    if verbose:
        print(f"  First: {ts[0]}  Last: {ts[-1]}  n_frames={len(frames)}  ok={ok}")
        if not ok:
            for i in bad_idx[:10]:
                print(f"    Δ[{i}] = {deltas[i]} (expected {expected})  {frames[i]} -> {frames[i+1]}")
    return ok, deltas, bad_idx

def check_temporal_integrity(split_name, data, frame_step_min, sample_clips=10):
    print(f"\n[Check 2] Temporal integrity — split: {split_name}")
    n = len(data)
    k = min(sample_clips, n)
    if n == 0:
        print("  (empty split)")
        return
    idxs = random.sample(range(n), k)
    bad = 0
    for i in idxs:
        print(f"- Clip idx {i}:")
        ok, _, _ = temporal_integrity(data[i], frame_step_min, verbose=True)
        bad += (0 if ok else 1)
    print(f"  Summary: {k - bad} OK / {k} sampled")

def check_cross_split_ar_uniqueness(train, cv, test):
    print("\n[Check 3] Cross-split AR uniqueness")
    def ars_of(data):
        s = set()
        for x in data:
            ar = x.get("ar") or x.get("AR") or x.get("active_region") or None
            if ar is not None:
                try:
                    s.add(int(ar))
                except Exception:
                    s.add(str(ar))
        return s

    ars = {
        "train": ars_of(train),
        "cv": ars_of(cv),
        "test": ars_of(test),
    }

    inter_train_cv = ars["train"] & ars["cv"]
    inter_train_test = ars["train"] & ars["test"]
    inter_cv_test = ars["cv"] & ars["test"]

    any_bad = False
    for name, inter in [
        ("train ∩ cv", inter_train_cv),
        ("train ∩ test", inter_train_test),
        ("cv ∩ test", inter_cv_test),
    ]:
        if inter:
            any_bad = True
            print(f"  ❌ Overlap in {name}: {sorted(list(inter))[:20]}{' ...' if len(inter)>20 else ''}")
        else:
            print(f"  ✅ No overlap in {name}")

    if not any_bad:
        print("  ✅ AR uniqueness across splits: PASSED")
    else:
        print("  ❌ AR uniqueness across splits: FAILED")

def distribution_report(split_name, data, outdir):
    print(f"\n[Check 4] Distribution report — split: {split_name}")
    labels = [x.get("label") for x in data]
    by_label = Counter(labels)
    print("  Per-class counts:")
    for lab, cnt in sorted(by_label.items(), key=lambda t: (-t[1], str(t[0]))):
        print(f"    {lab}: {cnt}")
    ar_counts = Counter([x.get("ar") for x in data])
    total = sum(ar_counts.values())
    n_ars = len(ar_counts)
    mean = total / n_ars if n_ars else 0.0
    per_ar_path = Path(outdir) / f"{split_name}_clips_per_AR.csv"
    per_cls_path = Path(outdir) / f"{split_name}_class_counts.csv"
    per_ar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(per_ar_path, "w") as f:
        f.write("ar,clips\n")
        for ar, cnt in sorted(ar_counts.items(), key=lambda t: int(t[0]) if str(t[0]).isdigit() else str(t[0])):
            f.write(f"{ar},{cnt}\n")
    with open(per_cls_path, "w") as f:
        f.write("label,count\n")
        for lab, cnt in sorted(by_label.items(), key=lambda t: t[0] if t[0] is not None else ""):
            f.write(f"{lab},{cnt}\n")
    if n_ars:
        values = list(ar_counts.values())
        mn = min(values); mx = max(values)
    else:
        mn = mx = 0
    print(f"  ARs: {n_ars}  total clips: {total}  clips/AR: min={mn} max={mx} mean={mean:.2f}")
    print(f"  Saved: {per_cls_path} and {per_ar_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True, help="Path to train JSON metadata")
    ap.add_argument("--cv", type=str, required=True, help="Path to cv/validation JSON metadata")
    ap.add_argument("--test", type=str, required=True, help="Path to test JSON metadata")
    ap.add_argument("--cadence-minutes", type=int, default=12, help="Minutes between consecutive raw images")
    ap.add_argument("--frame-step-minutes", type=int, default=12, help="Intended minutes between frames *within a clip*")
    ap.add_argument("--sample-clips", type=int, default=10, help="Number of random clips to sample per split")
    ap.add_argument("--outdir", type=str, default="reports", help="Directory to save CSV summaries")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    if args.frame_step_minutes % args.cadence_minutes != 0:
        print(f"WARNING: frame_step_minutes ({args.frame_step_minutes}) is not a multiple of cadence_minutes ({args.cadence_minutes}).")

    random.seed(args.seed)

    train = load_json(args.train).get("data", [])
    cv = load_json(args.cv).get("data", [])
    test = load_json(args.test).get("data", [])

    for name, data in [("train", train), ("cv", cv), ("test", test)]:
        check_temporal_integrity(name, data, frame_step_min=args.frame_step_minutes, sample_clips=args.sample_clips)

    check_cross_split_ar_uniqueness(train, cv, test)

    outdir = args.outdir
    for name, data in [("train", train), ("cv", cv), ("test", test)]:
        distribution_report(name, data, outdir)

if __name__ == "__main__":
    main()

