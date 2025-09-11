#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create AR-aware train/cv/test splits starting from original splits.

- Reads three JSON files (train/cv/test) each with {"data": [...]} items.
- Mixes all items and reassigns **entire AR groups** to one split only.
- Tries to match the original counts (targets) as closely as possible.
- Deterministic with --seed. Run multiple times with different seeds to get folds.

Output:
- <outdir>/train.seed_<SEED>.json
- <outdir>/cv.seed_<SEED>.json
- <outdir>/test.seed_<SEED>.json
- <outdir>/summary.seed_<SEED>.txt

"""

from __future__ import annotations
import os
import sys
import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

SPLITS = ("train", "cv", "test")


def load_split(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a 'data' list.")
    return data


def save_split(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"data": items}
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def summarize_by_label(items: List[dict]) -> Dict[str, int]:
    cnt = Counter()
    for it in items:
        lbl = str(it.get("label", ""))
        cnt[lbl] += 1
    return dict(cnt)


def group_by_ar(items: List[dict]) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for it in items:
        ar = it.get("ar", None)
        if ar is None:
            # If AR is missing, treat every such item as its own unique AR
            # using a stable synthetic id (e.g., its 'id' or filepath).
            # Prefer 'ar' to be present in your data, but this keeps things robust.
            synthetic = f"AR_MISSING::{it.get('id', it.get('filepath', 'unknown'))}"
            groups[synthetic].append(it)
        else:
            groups[str(ar)].append(it)
    return dict(groups)


def compute_targets(original_counts: Dict[str, int],
                    override: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    if override is None:
        return dict(original_counts)
    # If user provided overrides, fall back to original if a key is missing
    out = {}
    for k in SPLITS:
        out[k] = int(override.get(k, original_counts[k]))
    return out


def best_fit_partition(
    groups: Dict[str, List[dict]],
    targets: Dict[str, int],
    seed: int
) -> Dict[str, str]:
    """
    Assign each AR (group) to one split, attempting to match target counts.

    Heuristic:
      - Shuffle groups (for variety across seeds).
      - Sort groups by descending size (first-fit-decreasing-ish).
      - For each group, choose the split that results in the smallest
        absolute deviation from target, with a mild penalty for overshoot.
    Returns:
      mapping: ar_id -> chosen_split
    """
    rng = random.Random(seed)

    # Prepare.
    ar_sizes = {ar: len(items) for ar, items in groups.items()}
    ars = list(groups.keys())
    rng.shuffle(ars)
    # Large to small helps packing
    ars.sort(key=lambda g: ar_sizes[g], reverse=True)

    assigned: Dict[str, str] = {}
    current_counts = {s: 0 for s in SPLITS}

    def cost_if_assign(split: str, size: int) -> float:
        after = current_counts[split] + size
        # Base deviation from target
        dev = abs(after - targets[split])
        # Overshoot penalty (make it a bit more expensive than being under)
        overshoot = max(0, after - targets[split])
        penalty = 0.25 * overshoot  # tweakable
        return dev + penalty

    for ar in ars:
        size = ar_sizes[ar]
        # Tie-break with small random jitter to avoid always picking the same
        scored = sorted(
            ((cost_if_assign(s, size) + rng.random() * 1e-6, s) for s in SPLITS),
            key=lambda x: x[0],
        )
        chosen = scored[0][1]
        assigned[ar] = chosen
        current_counts[chosen] += size

    return assigned


def build_splits_from_assignment(
    all_groups: Dict[str, List[dict]],
    assignment: Dict[str, str]
) -> Dict[str, List[dict]]:
    out = {s: [] for s in SPLITS}
    for ar, items in all_groups.items():
        split = assignment[ar]
        out[split].extend(items)
    # Optional: stable order (by timestamp then id then filepath) for reproducibility/readability
    for s in SPLITS:
        out[s].sort(key=lambda x: (
            str(x.get("timestamp", "")),
            str(x.get("id", "")),
            str(x.get("filepath", "")),
        ))
    return out


def assert_group_integrity(splits: Dict[str, List[dict]]) -> Tuple[bool, Dict[str, str]]:
    """
    Check that each AR occurs in exactly one split.
    Returns (ok, ar_to_split) and raises AssertionError if violated.
    """
    where: Dict[str, str] = {}
    for split, items in splits.items():
        for it in items:
            ar = it.get("ar", None)
            if ar is None:
                ar = f"AR_MISSING::{it.get('id', it.get('filepath', 'unknown'))}"
            ar = str(ar)
            prev = where.get(ar)
            if prev is None:
                where[ar] = split
            elif prev != split:
                # Found AR across multiple splits
                return False, where
    return True, where


def make_summary_text(
    seed: int,
    original_counts: Dict[str, int],
    targets: Dict[str, int],
    splits: Dict[str, List[dict]]
) -> str:
    lines = []
    lines.append(f"=== AR-aware Split Summary (seed={seed}) ===")
    lines.append("")
    lines.append("Original sizes:")
    for s in SPLITS:
        lines.append(f"  {s:>5s}: {original_counts[s]}")
    lines.append("")
    lines.append("Targets (attempted):")
    for s in SPLITS:
        lines.append(f"  {s:>5s}: {targets[s]}")
    lines.append("")
    lines.append("Achieved sizes:")
    for s in SPLITS:
        lines.append(f"  {s:>5s}: {len(splits[s])}   (Δ={len(splits[s]) - targets[s]:+d})")
    lines.append("")

    # Label breakdowns (optional but handy)
    lines.append("Label breakdown per split:")
    for s in SPLITS:
        cnt = summarize_by_label(splits[s])
        total = sum(cnt.values())
        lines.append(f"  [{s}] total={total}")
        for k, v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            frac = v / total if total else 0.0
            lines.append(f"    - {k}: {v} ({frac:.2%})")
    lines.append("")

    # Integrity check
    ok, _ = assert_group_integrity(splits)
    lines.append(f"AR group integrity: {'OK' if ok else 'VIOLATION'}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Create AR-aware train/cv/test splits matching original sizes."
    )
    ap.add_argument("--train", required=True, type=Path, help="Path to original train.json")
    ap.add_argument("--cv", required=True, type=Path, help="Path to original cv.json")
    ap.add_argument("--test", required=True, type=Path, help="Path to original test.json")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--target-train", type=int, default=None, help="Override target train size")
    ap.add_argument("--target-cv", type=int, default=None, help="Override target cv size")
    ap.add_argument("--target-test", type=int, default=None, help="Override target test size")
    ap.add_argument("--allow-missing-ar", action="store_true",
                    help="If set, items without 'ar' are allowed (each becomes its own synthetic AR).")
    args = ap.parse_args()

    # Load original splits
    train_items = load_split(args.train)
    cv_items = load_split(args.cv)
    test_items = load_split(args.test)

    original_counts = {
        "train": len(train_items),
        "cv": len(cv_items),
        "test": len(test_items),
    }

    # Mix everything
    all_items = []
    all_items.extend(train_items)
    all_items.extend(cv_items)
    all_items.extend(test_items)

    # Basic validation
    if not args.allow_missing_ar:
        missing = [it for it in all_items if it.get("ar", None) is None]
        if missing:
            raise ValueError(
                f"Found {len(missing)} items without 'ar'. "
                f"Either fix the metadata or rerun with --allow-missing-ar."
            )

    # Group by AR
    groups = group_by_ar(all_items)

    # Targets
    override = None
    if any(v is not None for v in (args.target_train, args.target_cv, args.target_test)):
        override = {
            "train": args.target_train if args.target_train is not None else original_counts["train"],
            "cv": args.target_cv if args.target_cv is not None else original_counts["cv"],
            "test": args.target_test if args.target_test is not None else original_counts["test"],
        }
    targets = compute_targets(original_counts, override)

    # Assignment
    assignment = best_fit_partition(groups, targets, seed=args.seed)
    new_splits = build_splits_from_assignment(groups, assignment)

    # Integrity check
    ok, ar_map = assert_group_integrity(new_splits)
    if not ok:
        raise AssertionError("AR group integrity violated: an AR appears in multiple splits.")

    # Write outputs
    args.outdir.mkdir(parents=True, exist_ok=True)
    #out_train = args.outdir / f"train.seed_{args.seed}.json"
    #out_cv = args.outdir / f"cv.seed_{args.seed}.json"
    #out_test = args.outdir / f"test.seed_{args.seed}.json"
    filename_train= os.path.basefile(args.train) 
    filename_cv= os.path.basefile(args.cv) 
    filename_test= os.path.basefile(args.test) 
    out_train = args.outdir / f"train/{filename_train}.json"
    out_cv = args.outdir / f"cv/{filename_cv}.json"
    out_test = args.outdir / f"test/{filename_test}.json"
    
    save_split(out_train, new_splits["train"])
    save_split(out_cv, new_splits["cv"])
    save_split(out_test, new_splits["test"])

    # Write summary
    summary = make_summary_text(args.seed, original_counts, targets, new_splits)
    summary_path = args.outdir / f"summary.seed_{args.seed}.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print(f"\nWrote:\n  {out_train}\n  {out_cv}\n  {out_test}\n  {summary_path}")


if __name__ == "__main__":
    main()

