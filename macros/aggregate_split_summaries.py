#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate AR-aware split summary files across multiple randomized runs.

- Recursively scans a top directory for files matching a glob pattern
  (default: **/summary.seed_*.txt).
- Parses per-split, per-label composition tables (NEW data) from each summary:
    [train] Per-label composition (original vs new):
      label  | ... |  new_n  new_% | ...
    [train] Collapsed composition (NONE vs M+ vs OTHER):
      bucket | ... |  new_n  new_% | ...
- Computes min, max, mean, median, std for counts and fractions across runs.
- Writes a single CSV with columns:
    category,split,label,metric,n,min,max,mean,median,std

Usage:
    python aggregate_split_summaries.py /path/to/topdir \
        --pattern "summary.seed_*.txt" \
        --out stats.csv \
        [--missing-as-zero]

Notes:
- Assumes summary format produced by the patched splitter we discussed.
- If the "Class composition comparison" block is missing, a fallback parser
  will try the "Label breakdown per NEW split" block.
"""

from __future__ import annotations
import argparse
import csv
import os
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple, Optional, Any, DefaultDict
from collections import defaultdict

SPLITS = ("train", "cv", "test")

# ---------------------------
# Parsing helpers
# ---------------------------

# Regex for table rows in the "Per-label composition (original vs new)" blocks:
# Example row:
# "  NONE                  |   9987   79.90% |   9987    79.90% |   -13   -0.10%"
ROW_RE = re.compile(
    r"^\s*(?P<label>.+?)\s*\|\s*"
    r"(?P<on>\d+)\s+(?P<of>[0-9.]+)%\s*\|\s*"
    r"(?P<nn>\d+)\s+(?P<nf>[0-9.]+)%\s*\|\s*"
    r"(?P<dn>[+\-]?\d+)\s+(?P<df>[+\-]?[0-9.]+)%",
    re.ASCII
)

HEADER_PER_LABEL_RE = re.compile(
    r"^\[(?P<split>train|cv|test)\]\s+Per-label composition",
    re.IGNORECASE
)

HEADER_COLLAPSED_RE = re.compile(
    r"^\[(?P<split>train|cv|test)\]\s+Collapsed composition",
    re.IGNORECASE
)

# Fallback parser for:
# "Label breakdown per NEW split:" then blocks like:
#   [train] total=12345
#     - LABEL: 999 (12.34%)
FALLBACK_SECTION_RE = re.compile(r"^Label breakdown per NEW split:", re.IGNORECASE)
FALLBACK_SPLIT_RE = re.compile(r"^\s*\[(?P<split>train|cv|test)\]\s+total\s*=\s*(?P<total>\d+)", re.IGNORECASE)
FALLBACK_ROW_RE = re.compile(
    r"^\s*-\s*(?P<label>.+?)\s*:\s*(?P<n>\d+)\s*\((?P<pct>[0-9.]+)%\)",
    re.ASCII
)

def parse_summary_file(path: Path) -> Dict[str, Dict[str, Dict[str, Tuple[int, float]]]]:
    """
    Parse a single summary file.

    Returns a nested dict:
      {
        'per_label': { split: { label: (new_n, new_frac) } },
        'collapsed': { split: { label: (new_n, new_frac) } }   # if present
      }
    Unknown or missing blocks are simply omitted.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    out: Dict[str, Dict[str, Dict[str, Tuple[int, float]]]] = {
        "per_label": {s: {} for s in SPLITS},
        "collapsed": {s: {} for s in SPLITS},
    }
    seen_any = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Per-label table
        m = HEADER_PER_LABEL_RE.match(line)
        if m:
            split = m.group("split").lower()
            i += 1  # move to header separator
            # Skip the two header lines (separator)
            # We will read rows until a blank line or next header
            # First, try to skip up to 2 lines if they look like headers
            skip = 0
            while i < len(lines) and skip < 3 and "|" in lines[i]:
                # consume header lines (there are usually 2)
                i += 1
                skip += 1

            # Now parse rows
            while i < len(lines) and lines[i].strip():
                rm = ROW_RE.match(lines[i])
                if not rm:
                    break
                lab = rm.group("label").strip()
                nn = int(rm.group("nn"))
                nf = float(rm.group("nf")) / 100.0
                out["per_label"][split][lab] = (nn, nf)
                seen_any = True
                i += 1
            continue

        # Collapsed table
        m = HEADER_COLLAPSED_RE.match(line)
        if m:
            split = m.group("split").lower()
            i += 1
            # Skip the header lines (similar approach)
            skip = 0
            while i < len(lines) and skip < 3 and "|" in lines[i]:
                i += 1
                skip += 1
            while i < len(lines) and lines[i].strip():
                rm = ROW_RE.match(lines[i])
                if not rm:
                    break
                lab = rm.group("label").strip()
                nn = int(rm.group("nn"))
                nf = float(rm.group("nf")) / 100.0
                out["collapsed"][split][lab] = (nn, nf)
                seen_any = True
                i += 1
            continue

        i += 1

    # Fallback block if per-label wasn’t found
    if not seen_any:
        out = {"per_label": {s: {} for s in SPLITS}, "collapsed": {s: {}}}
        in_section = False
        cur_split = None
        totals = {s: None for s in SPLITS}
        for line in lines:
            if FALLBACK_SECTION_RE.search(line):
                in_section = True
                cur_split = None
                continue
            if not in_section:
                continue

            m = FALLBACK_SPLIT_RE.match(line)
            if m:
                cur_split = m.group("split").lower()
                totals[cur_split] = int(m.group("total"))
                continue

            if cur_split:
                rm = FALLBACK_ROW_RE.match(line.strip())
                if rm:
                    lab = rm.group("label").strip()
                    n = int(rm.group("n"))
                    pct = float(rm.group("pct")) / 100.0
                    out["per_label"][cur_split][lab] = (n, pct)

    return out

# ---------------------------
# Aggregation
# ---------------------------

def safe_stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)

def aggregate_runs(
    runs: List[Dict[str, Dict[str, Dict[str, Tuple[int, float]]]]],
    missing_as_zero: bool = False
) -> List[Dict[str, Any]]:
    """
    Aggregate across runs.
    Returns a list of rows dicts ready for CSV writing:
      {
        "category": "per_label" | "collapsed",
        "split": "train" | "cv" | "test",
        "label": "<label>",
        "metric": "count" | "frac",
        "n": int,          # number of runs included
        "min": float,
        "max": float,
        "mean": float,
        "median": float,
        "std": float
      }
    """
    results: List[Dict[str, Any]] = []

    for category in ("per_label", "collapsed"):
        for split in SPLITS:
            # Build the union of labels across runs
            label_union = set()
            for r in runs:
                label_union |= set(r.get(category, {}).get(split, {}).keys())

            for label in sorted(label_union):
                counts: List[float] = []
                fracs:  List[float] = []
                for r in runs:
                    d = r.get(category, {}).get(split, {})
                    if label in d:
                        n, f = d[label]
                        counts.append(float(n))
                        fracs.append(float(f))
                    else:
                        if missing_as_zero:
                            counts.append(0.0)
                            fracs.append(0.0)
                        # else: skip this run for this label

                if counts:
                    rows = [
                        {
                            "category": category,
                            "split": split,
                            "label": label,
                            "metric": "count",
                            "n": len(counts),
                            "min": min(counts),
                            "max": max(counts),
                            "mean": mean(counts),
                            "median": median(counts),
                            "std": safe_stdev(counts),
                        },
                        {
                            "category": category,
                            "split": split,
                            "label": label,
                            "metric": "frac",
                            "n": len(fracs),
                            "min": min(fracs),
                            "max": max(fracs),
                            "mean": mean(fracs),
                            "median": median(fracs),
                            "std": safe_stdev(fracs),
                        },
                    ]
                    results.extend(rows)

    return results

# ---------------------------
# I/O and CLI
# ---------------------------

def find_summary_files(top: Path, pattern: str) -> List[Path]:
    # pattern is a glob like "summary.seed_*.txt"
    files = []
    for p in top.rglob(pattern):
        if p.is_file():
            files.append(p)
    # Sort for stable ordering
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser(description="Aggregate split summary stats across randomized runs.")
    ap.add_argument("topdir", type=Path, help="Top directory to search recursively.")
    ap.add_argument("--pattern", type=str, default="summary.seed_*.txt",
                    help="Glob pattern for summary files (default: summary.seed_*.txt).")
    ap.add_argument("--out", type=Path, default=Path("split_stats.csv"),
                    help="Output CSV path (default: split_stats.csv).")
    ap.add_argument("--missing-as-zero", action="store_true",
                    help="Treat missing labels as zero instead of skipping those runs.")
    args = ap.parse_args()

    files = find_summary_files(args.topdir, args.pattern)
    if not files:
        raise SystemExit(f"No summary files found under {args.topdir} matching pattern '{args.pattern}'.")

    runs = []
    for f in files:
        try:
            runs.append(parse_summary_file(f))
        except Exception as e:
            print(f"⚠️  Skipping {f}: parse error: {e}")

    if not runs:
        raise SystemExit("No parsable summary files.")

    rows = aggregate_runs(runs, missing_as_zero=args.missing_as_zero)

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "category", "split", "label", "metric", "n", "min", "max", "mean", "median", "std"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Processed {len(runs)} summaries from {len(files)} files.")
    print(f"Wrote stats to: {args.out}")

if __name__ == "__main__":
    main()

