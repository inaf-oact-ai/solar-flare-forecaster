#!/usr/bin/env python3
"""
Split a JSON metadata file into train / cv / test splits with:
  • Reproducibility via seed
  • Stratified splitting by class (default key: 'flare_type')
  • Optional class balancing of the TRAIN set with configurable targets
    - Downsample (default)
    - Oversample by duplicating entries in the train JSON (optional)

Input JSON schema (minimal):
{
    "data": [
        { "flare_type": "NONE", ... other fields ... },
        { "flare_type": "C", ... },
        ...
    ]
}

Outputs:
  out_dir/train.json
  out_dir/cv.json
  out_dir/test.json
  out_dir/summary.txt  (human-readable stats)

Examples
--------
1) Plain 70/10/20 split, stratified by flare_type:
   python split_dataset_with_train_class_balance.py \
       --input meta.json --out-dir splits --seed 42 \
       --fractions 0.7 0.1 0.2

2) Make train more balanced (downsample to fixed counts):
   python split_dataset_with_train_class_balance.py \
       --input meta.json --out-dir splits --seed 123 \
       --fractions 0.7 0.1 0.2 \
       --train-target-per-class NONE=500 C=500 M=500 X=500 \
       --train-balance-strategy downsample

3) If you really want to oversample within train JSON (duplicates allowed):
   python split_dataset_with_train_class_balance.py \
       --input meta.json --out-dir splits --seed 7 \
       --fractions 0.7 0.1 0.2 \
       --train-target-per-class NONE=1000 C=800 M=600 X=400 \
       --train-balance-strategy oversample_in_metadata

Notes
-----
• If --train-target-per-class is omitted, the script does a standard stratified split with the given fractions.
• With downsampling, train never exceeds available samples per class; with oversample_in_metadata, the train set may include duplicate entries to reach the requested targets.
• CV and test are always stratified using the remaining (not used) unique items; they never include duplicates or overlap with train.
"""

from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any


def parse_train_targets(pairs: List[str]) -> Dict[str, int]:
    """Parse key=value pairs like ["NONE=500", "C=500"]."""
    targets: Dict[str, int] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid target spec '{p}'. Use CLASS=COUNT.")
        k, v = p.split("=", 1)
        k = k.strip()
        try:
            targets[k] = int(v)
        except ValueError:
            raise ValueError(f"Invalid count in '{p}': must be an integer.")
        if targets[k] < 0:
            raise ValueError(f"Count for class '{k}' must be >= 0")
    return targets


def load_metadata(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "data" not in obj or not isinstance(obj["data"], list):
        raise ValueError("Input JSON must be an object with a 'data' list")
    return obj["data"]


def write_metadata(path: Path, items: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"data": items}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def stratified_take(indices_by_class: Dict[str, List[int]], counts_by_class: Dict[str, int]) -> List[int]:
    """Take (without replacement) the requested number from each class list (which should be pre-shuffled)."""
    taken: List[int] = []
    for cls, need in counts_by_class.items():
        pool = indices_by_class.get(cls, [])
        if need > len(pool):
            need = len(pool)
        taken.extend(pool[:need])
        # Shrink pool to reflect consumption
        indices_by_class[cls] = pool[need:]
    return taken


def compute_split_counts(n_total: int, train_frac: float, cv_frac: float, test_frac: float) -> Tuple[int, int, int]:
    # Robust rounding so the three parts sum exactly to n_total
    train = int(round(n_total * train_frac))
    cv = int(round(n_total * cv_frac))
    test = n_total - train - cv
    # Adjust if rounding led to negative or overflow
    if test < 0:
        # pull from cv first
        deficit = -test
        take_from_cv = min(deficit, cv)
        cv -= take_from_cv
        deficit -= take_from_cv
        train -= deficit
        test = 0
    return train, cv, test


def summarize_split(name: str, items: List[dict], class_key: str) -> str:
    cnt = Counter([it.get(class_key) for it in items])
    lines = [f"{name}: n={len(items)}"]
    for k in sorted(cnt.keys(), key=lambda x: (str(x))):
        lines.append(f"  - {k}: {cnt[k]}")
    return "\n".join(lines)


def make_splits(
    data: List[dict],
    class_key: str,
    seed: int,
    train_frac: float,
    cv_frac: float,
    test_frac: float,
    train_targets: Dict[str, int] | None,
    balance_strategy: str,
    classes_order: List[str] | None,
) -> Tuple[List[dict], List[dict], List[dict], str]:
    rng = random.Random(seed)

    # Index data by class
    indices_by_class: Dict[str, List[int]] = defaultdict(list)
    for idx, it in enumerate(data):
        cls = it.get(class_key)
        indices_by_class[cls].append(idx)

    # Optional explicit class order for deterministic summaries
    all_classes = list(indices_by_class.keys()) if not classes_order else list(classes_order)

    # Shuffle each class pool reproducibly
    for cls in indices_by_class:
        rng.shuffle(indices_by_class[cls])

    n_total = len(data)
    train_n, cv_n, test_n = compute_split_counts(n_total, train_frac, cv_frac, test_frac)

    # If balancing is requested, we decide TRAIN counts per class explicitly.
    if train_targets:
        # Start with requested targets; cap to available if downsampling only
        per_class_train = {}
        for cls in all_classes:
            target = train_targets.get(cls, 0)
            available = len(indices_by_class.get(cls, []))
            if balance_strategy == "downsample":
                per_class_train[cls] = min(target, available)
            elif balance_strategy == "oversample_in_metadata":
                per_class_train[cls] = target
            else:
                raise ValueError("balance_strategy must be 'downsample' or 'oversample_in_metadata'")
        # If the sum of requested train counts exceeds the overall train budget, warn by trimming proportionally.
        total_requested = sum(per_class_train.values())
        if total_requested > train_n and balance_strategy == "downsample":
            # Scale down proportionally to fit the train budget
            scale = train_n / total_requested if total_requested > 0 else 0.0
            for cls in per_class_train:
                per_class_train[cls] = int(math.floor(per_class_train[cls] * scale))
        # Take unique items for train from each class pool
        taken_train_indices = stratified_take(indices_by_class, per_class_train)
        train_items = [data[i] for i in taken_train_indices]

        # Handle oversampling by duplicating entries to reach target counts
        if balance_strategy == "oversample_in_metadata":
            # Count how many per class we already have in train_items
            have = Counter([it.get(class_key) for it in train_items])
            augmented: List[dict] = list(train_items)
            for cls in all_classes:
                want = per_class_train.get(cls, 0)
                got = have.get(cls, 0)
                if want > got:
                    pool = [data[i] for i in indices_by_class.get(cls, [])] + [it for it in train_items if it.get(class_key) == cls]
                    # If pool is empty (no samples of that class exist), skip
                    if pool:
                        need = want - got
                        for _ in range(need):
                            augmented.append(rng.choice(pool))
            train_items = augmented

        # Now split the REMAINING unique indices into cv/test stratified by class pools left
        remaining_indices = []
        for cls in indices_by_class:
            remaining_indices.extend(indices_by_class[cls])
        rng.shuffle(remaining_indices)
        # Build remaining pools per class again (already shuffled above, but we consumed heads)
        rem_by_class: Dict[str, List[int]] = {}
        for cls in indices_by_class:
            rem_by_class[cls] = list(indices_by_class[cls])

        # Compute target cv/test counts per class proportional to what's left
        total_remaining = sum(len(v) for v in rem_by_class.values())
        cv_counts, test_counts = {}, {}
        for cls, pool in rem_by_class.items():
            n_cls = len(pool)
            if total_remaining > 0:
                cv_counts[cls] = int(round(n_cls * (cv_n / total_remaining)))
            else:
                cv_counts[cls] = 0
        # Take CV
        taken_cv = stratified_take(rem_by_class, cv_counts)
        cv_items = [data[i] for i in taken_cv]

        # Remaining go to TEST (respecting test_n by truncation if needed)
        remaining_after_cv = []
        for cls in rem_by_class:
            remaining_after_cv.extend(rem_by_class[cls])
        rng.shuffle(remaining_after_cv)
        taken_test = remaining_after_cv[:test_n]
        test_items = [data[i] for i in taken_test]

    else:
        # No balancing targets: do a standard stratified split across all sets.
        # Decide class-wise quotas proportionally to class sizes.
        total_counts = {cls: len(indices_by_class[cls]) for cls in indices_by_class}
        total_sum = sum(total_counts.values())

        train_quota, cv_quota, test_quota = {}, {}, {}
        for cls, n_cls in total_counts.items():
            train_quota[cls] = int(round(n_cls * (train_n / total_sum))) if total_sum else 0
            cv_quota[cls] = int(round(n_cls * (cv_n / total_sum))) if total_sum else 0
            # test will fill the rest
        # Take TRAIN
        taken_train = stratified_take(indices_by_class, train_quota)
        train_items = [data[i] for i in taken_train]
        # Take CV
        taken_cv = stratified_take(indices_by_class, cv_quota)
        cv_items = [data[i] for i in taken_cv]
        # Remaining -> TEST
        remaining = []
        for cls in indices_by_class:
            remaining.extend(indices_by_class[cls])
        rng.shuffle(remaining)
        taken_test = remaining[:test_n]
        test_items = [data[i] for i in taken_test]

    # Build a readable summary
    summary_lines = []
    summary_lines.append("=== SPLIT SUMMARY ===")
    summary_lines.append(f"Total items: {n_total}")
    summary_lines.append("")
    summary_lines.append(summarize_split("TRAIN", train_items, class_key))
    summary_lines.append("")
    summary_lines.append(summarize_split("CV", cv_items, class_key))
    summary_lines.append("")
    summary_lines.append(summarize_split("TEST", test_items, class_key))

    return train_items, cv_items, test_items, "\n".join(summary_lines)


def main():
    ap = argparse.ArgumentParser(description="Create reproducible stratified splits with optional train balancing.")
    ap.add_argument("--input", required=True, type=Path, help="Path to input metadata JSON")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory for split JSON files")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--fractions", nargs=3, type=float, metavar=("TRAIN", "CV", "TEST"), default=[0.7, 0.1, 0.2],
                    help="Fractions for train/cv/test; must sum to ~1.0")
    ap.add_argument("--class-key", type=str, default="flare_type", help="JSON field name for the class label")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="Optional explicit list of classes to order summaries and targets (e.g., NONE C M X)")
    ap.add_argument("--train-target-per-class", nargs="*", default=None,
                    help="Optional per-class train targets like NONE=500 C=500 M=500 X=500")
    ap.add_argument("--train-balance-strategy", choices=["downsample", "oversample_in_metadata"], default="downsample",
                    help="How to enforce train targets: downsample (no duplicates) or oversample_in_metadata (allow duplicates)")

    args = ap.parse_args()

    train_frac, cv_frac, test_frac = args.fractions
    if not (0 <= train_frac <= 1 and 0 <= cv_frac <= 1 and 0 <= test_frac <= 1):
        raise SystemExit("Fractions must be between 0 and 1")
    if not math.isclose(train_frac + cv_frac + test_frac, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise SystemExit("Fractions must sum to 1.0")

    train_targets = parse_train_targets(args.train_target_per_class) if args.train_target_per_class else None

    data = load_metadata(args.input)

    train_items, cv_items, test_items, summary = make_splits(
        data=data,
        class_key=args.class_key,
        seed=args.seed,
        train_frac=train_frac,
        cv_frac=cv_frac,
        test_frac=test_frac,
        train_targets=train_targets,
        balance_strategy=args.train_balance_strategy,
        classes_order=args.classes,
    )

    # Write outputs
    out_dir: Path = args.out_dir
    write_metadata(out_dir / "train.json", train_items)
    write_metadata(out_dir / "cv.json", cv_items)
    write_metadata(out_dir / "test.json", test_items)

    # Summary file
    (out_dir / "summary.txt").parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(summary)
    print(f"\nWrote: {out_dir / 'train.json'}")
    print(f"Wrote: {out_dir / 'cv.json'}")
    print(f"Wrote: {out_dir / 'test.json'}")
    print(f"Wrote: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
