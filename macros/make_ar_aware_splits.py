#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        cnt[str(it.get("label", ""))] += 1
    return dict(cnt)

def group_by_ar(items: List[dict]) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for it in items:
        ar = it.get("ar", None)
        if ar is None:
            ar = f"AR_MISSING::{it.get('id', it.get('filepath', 'unknown'))}"
        groups[str(ar)].append(it)
    return dict(groups)

def compute_targets(original_counts: Dict[str, int],
                    override: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    if override is None:
        return dict(original_counts)
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
    Assign each AR (group) to a split, prioritizing filling under-target splits first.
    Then use cost (deviation + mild overshoot penalty) to decide among candidates.
    """
    rng = random.Random(seed)
    ar_sizes = {ar: len(items) for ar, items in groups.items()}

    ars = list(groups.keys())
    rng.shuffle(ars)
    ars.sort(key=lambda g: ar_sizes[g], reverse=True)  # big first helps packing

    assigned: Dict[str, str] = {}
    current = {s: 0 for s in SPLITS}

    def cost(split: str, size: int) -> float:
        after = current[split] + size
        dev = abs(after - targets[split])
        overshoot = max(0, after - targets[split])
        return dev + 0.25 * overshoot + rng.random() * 1e-6  # tiny jitter

    for ar in ars:
        size = ar_sizes[ar]
        # 1) Prefer splits still under target
        under = [s for s in SPLITS if current[s] < targets[s]]
        candidates = under if under else list(SPLITS)
        # 2) Pick cheapest among candidates
        chosen = min(candidates, key=lambda s: cost(s, size))
        assigned[ar] = chosen
        current[chosen] += size

    return assigned

def rebalance_assignment(
    groups: Dict[str, List[dict]],
    assignment: Dict[str, str],
    targets: Dict[str, int],
    max_iters: int = 200
) -> Dict[str, str]:
    """
    Greedy post-pass: if a split is under target and another is over, move the
    **smallest** AR from an overfull split into the most underfull split, if it improves total L1 deviation.
    """
    ar_sizes = {ar: len(items) for ar, items in groups.items()}

    def counts(asg: Dict[str, str]) -> Dict[str, int]:
        c = {s: 0 for s in SPLITS}
        for ar, sp in asg.items():
            c[sp] += ar_sizes[ar]
        return c

    def total_l1(cnts: Dict[str, int]) -> int:
        return sum(abs(cnts[s] - targets[s]) for s in SPLITS)

    asg = dict(assignment)
    for _ in range(max_iters):
        cnts = counts(asg)
        l1 = total_l1(cnts)

        # identify most under & most over
        most_under = min(SPLITS, key=lambda s: cnts[s] - targets[s])  # most negative delta
        most_over  = max(SPLITS, key=lambda s: cnts[s] - targets[s])  # most positive delta

        under_def = targets[most_under] - cnts[most_under]
        over_excess = cnts[most_over] - targets[most_over]

        if under_def <= 0 or over_excess <= 0:
            break  # balanced enough or no improving move possible

        # candidate ARs to move: those currently in most_over; try smallest first
        cands = [ar for ar, sp in asg.items() if sp == most_over]
        if not cands:
            break
        cands.sort(key=lambda ar: ar_sizes[ar])  # move the smallest that helps

        moved = False
        for ar in cands:
            size = ar_sizes[ar]
            # Try move
            asg[ar] = most_under
            new_cnts = counts(asg)
            if total_l1(new_cnts) < l1:
                moved = True
                break
            # revert if no improvement
            asg[ar] = most_over

        if not moved:
            break  # no improving move found
    return asg

def build_splits_from_assignment(
    all_groups: Dict[str, List[dict]],
    assignment: Dict[str, str]
) -> Dict[str, List[dict]]:
    out = {s: [] for s in SPLITS}
    for ar, items in all_groups.items():
        out[assignment[ar]].extend(items)
    for s in SPLITS:
        out[s].sort(key=lambda x: (
            str(x.get("timestamp", "")),
            str(x.get("id", "")),
            str(x.get("filepath", "")),
        ))
    return out

def assert_group_integrity(splits: Dict[str, List[dict]]) -> Tuple[bool, Dict[str, str]]:
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
                return False, where
    return True, where

def make_summary_text(
    seed: int,
    original_counts: Dict[str, int],
    targets: Dict[str, int],
    splits: Dict[str, List[dict]]
) -> str:
    lines = []
    lines.append(f"=== AR-aware Split Summary (seed={seed}) ===\n")
    lines.append("Original sizes:")
    for s in SPLITS:
        lines.append(f"  {s:>5s}: {original_counts[s]}")
    lines.append("\nTargets (attempted):")
    for s in SPLITS:
        lines.append(f"  {s:>5s}: {targets[s]}")
    lines.append("\nAchieved sizes:")
    for s in SPLITS:
        lines.append(f"  {s:>5s}: {len(splits[s])}   (Δ={len(splits[s]) - targets[s]:+d})")
    lines.append("")
    lines.append("Label breakdown per split:")
    for s in SPLITS:
        cnt = summarize_by_label(splits[s])
        total = sum(cnt.values())
        lines.append(f"  [{s}] total={total}")
        for k, v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            frac = v / total if total else 0.0
            lines.append(f"    - {k}: {v} ({frac:.2%})")
    lines.append("")
    ok, _ = assert_group_integrity(splits)
    lines.append(f"AR group integrity: {'OK' if ok else 'VIOLATION'}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Create AR-aware train/cv/test splits matching original sizes.")
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--cv", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-train", type=int, default=None)
    ap.add_argument("--target-cv", type=int, default=None)
    ap.add_argument("--target-test", type=int, default=None)
    ap.add_argument("--allow-missing-ar", action="store_true")
    args = ap.parse_args()

    train_items = load_split(args.train)
    cv_items = load_split(args.cv)
    test_items = load_split(args.test)

    original_counts = {"train": len(train_items), "cv": len(cv_items), "test": len(test_items)}
    all_items = train_items + cv_items + test_items

    if not args.allow_missing_ar:
        missing = [it for it in all_items if it.get("ar", None) is None]
        if missing:
            raise ValueError(
                f"Found {len(missing)} items without 'ar'. "
                f"Fix metadata or rerun with --allow-missing-ar."
            )

    groups = group_by_ar(all_items)

    override = None
    if any(v is not None for v in (args.target_train, args.target_cv, args.target_test)):
        override = {
            "train": args.target_train if args.target_train is not None else original_counts["train"],
            "cv": args.target_cv if args.target_cv is not None else original_counts["cv"],
            "test": args.target_test if args.target_test is not None else original_counts["test"],
        }
    targets = compute_targets(original_counts, override)

    # Initial assignment + rebalance
    assignment0 = best_fit_partition(groups, targets, seed=args.seed)
    assignment  = rebalance_assignment(groups, assignment0, targets)

    new_splits = build_splits_from_assignment(groups, assignment)
    ok, _ = assert_group_integrity(new_splits)
    if not ok:
        raise AssertionError("AR group integrity violated: an AR appears in multiple splits.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    filename_train= (args.train).name 
    filename_cv= (args.cv).name 
    filename_test= (args.test).name 
    out_train = args.outdir / f"train/{filename_train}.json"
    out_cv = args.outdir / f"cv/{filename_cv}.json"
    out_test = args.outdir / f"test/{filename_test}.json"
    save_split(out_train, new_splits["train"])
    save_split(out_cv,    new_splits["cv"])
    save_split(out_test,  new_splits["test"])

    summary = make_summary_text(args.seed, original_counts, targets, new_splits)
    summary_path = args.outdir / f"summary.seed_{args.seed}.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    print(f"\nWrote:\n  {out_train}\n  {out_cv}\n  {out_test}\n  {summary_path}")

if __name__ == "__main__":
    main()











    
