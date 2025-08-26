#!/usr/bin/env python3
"""
balance_none_by_ar.py

Downsample "NONE" entries in an HMI image JSON dataset, stratified by AR (active region).

Input format:
{
  "data": [
    {
      "filepath": "...",
      "sname": "...",
      "label": "C" | "M" | "X" | "NONE",
      "id": 1,
      "ar": 1069,
      "timestamp": "2010-05-05 00:00:00"
    },
    ...
  ]
}

You can specify either:
  --target-total <N>   : desired total number of entries in the output
  --target-none  <N>   : desired number of "NONE" entries in the output

Only "NONE" entries are removed. Non-"NONE" entries are always kept.

Removal is stratified by "ar" so that small-AR groups are not fully eliminated unless mathematically
unavoidable (e.g., you need fewer "NONE" entries than the number of AR groups).

Example usages:
  python balance_none_by_ar.py --input in.json --output out.json --target-total 50000 --seed 42
  python balance_none_by_ar.py --input in.json --output out.json --target-none 12000 --removed removed.json
"""

import argparse
import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

LABEL_NONE = "NONE"


def parse_args():
    p = argparse.ArgumentParser(description="Downsample 'NONE' entries stratified by AR.")
    p.add_argument("--input", required=True, help="Path to input JSON file")
    p.add_argument("--output", required=True, help="Path to output JSON file")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--target-total", type=int, help="Desired total number of entries in output")
    grp.add_argument("--target-none", type=int, help="Desired number of NONE entries in output")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--removed", default=None, help="Optional path to save removed entries as JSON")
    p.add_argument("--dry-run", action="store_true", help="Compute plan and print summary; do not write files")
    return p.parse_args()


def load_data(path: str) -> List[dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    if "data" not in payload or not isinstance(payload["data"], list):
        raise ValueError("Input JSON must contain a top-level 'data' list.")
    return payload["data"]


def save_data(path: str, data: List[dict]):
    with open(path, "w") as f:
        json.dump({"data": data}, f, ensure_ascii=False, indent=2)


def group_none_by_ar(records: List[dict]) -> Dict[int, List[int]]:
    """
    Returns dict: ar -> list of indices (positions in 'records') for NONE entries in that AR.
    """
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        if rec.get("label") == LABEL_NONE:
            ar = rec.get("ar")
            groups[ar].append(idx)
    return groups


def compute_removal_plan(
    none_groups: Dict[int, List[int]],
    keep_total_none: int,
    rng: random.Random
) -> Tuple[Dict[int, int], int]:
    """
    Given NONE groups (ar -> indices), and the desired total NONE to keep,
    compute how many to keep per AR (then removals are implied).
    Strategy:
      1) If keep_total_none >= #groups: guarantee at least 1 kept per group (when possible),
         then distribute remaining proportionally to group sizes (via randomized rounding).
      2) If keep_total_none < #groups: choose keep_total_none groups to keep (1 each),
         sampled with probability proportional to group size; other groups keep 0.
    Returns: (keep_per_ar, actually_kept)
    """
    ars = list(none_groups.keys())
    sizes = {ar: len(none_groups[ar]) for ar in ars}
    G = len(ars)
    total_none = sum(sizes.values())

    # Clamp the desired total to feasible range
    keep_total_none = max(0, min(keep_total_none, total_none))
    keep_per_ar = {ar: 0 for ar in ars}

    if keep_total_none == 0 or G == 0:
        return keep_per_ar, 0

    if keep_total_none >= G:
        # Start by giving one to each non-empty group
        base_allocate = sum(1 for ar in ars if sizes[ar] > 0)
        for ar in ars:
            if sizes[ar] > 0:
                keep_per_ar[ar] = 1

        remaining = keep_total_none - base_allocate
        if remaining > 0:
            # Distribute remaining proportionally to residual capacities
            weights = {ar: max(0, sizes[ar] - keep_per_ar[ar]) for ar in ars}
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                ideal = {ar: (weights[ar] / weight_sum) * remaining for ar in ars}
                floors = {ar: int(math.floor(ideal[ar])) for ar in ars}
                allocated = sum(floors.values())
                for ar in ars:
                    keep_per_ar[ar] += floors[ar]

                leftover = remaining - allocated
                if leftover > 0:
                    remainders = {ar: ideal[ar] - floors[ar] for ar in ars}
                    rem_sum = sum(remainders.values())
                    probs = [remainders[ar] if rem_sum > 0 else weights[ar] for ar in ars]
                    chosen = weighted_sample_without_replacement(ars, probs, leftover, rng)
                    for ar in chosen:
                        keep_per_ar[ar] += 1

        # Cap by group sizes
        for ar in ars:
            keep_per_ar[ar] = min(keep_per_ar[ar], sizes[ar])

    else:
        # keep_total_none < number of groups
        probs = [sizes[ar] for ar in ars]
        chosen_groups = set(weighted_sample_without_replacement(ars, probs, keep_total_none, rng))
        for ar in chosen_groups:
            keep_per_ar[ar] = 1

    actually_kept = sum(keep_per_ar.values())
    return keep_per_ar, actually_kept


def weighted_sample_without_replacement(items: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
    """
    Efraimidis-Spirakis method for weighted sampling without replacement.
    """
    assert len(items) == len(weights)
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)

    keys = []
    for item, w in zip(items, weights):
        w = max(0.0, float(w))
        if w == 0.0:
            key = float("-inf")
        else:
            u = rng.random()
            u = max(u, 1e-12)  # avoid log(0)
            key = math.log(u) / w
        keys.append((key, item))
    keys.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in keys[:k]]


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    records = load_data(args.input)

    # Split by label
    none_indices = [i for i, r in enumerate(records) if r.get("label") == LABEL_NONE]
    non_none_indices = [i for i, r in enumerate(records) if r.get("label") != LABEL_NONE]

    n_total = len(records)
    n_none = len(none_indices)
    n_non_none = len(non_none_indices)

    # Resolve targets
    if args.target_none is not None:
        target_none = args.target_none
        if target_none < 0:
            raise ValueError("--target-none must be non-negative")
        if target_none > n_none:
            print(f"[INFO] Requested target-none={target_none} > current NONE={n_none}. No removals performed on NONEs.")
            target_none = n_none
        remove_none = n_none - target_none
        target_total = n_non_none + target_none
    else:
        # --target-total was provided
        target_total = args.target_total
        if target_total < n_non_none:
            raise ValueError(
                f"--target-total={target_total} is smaller than non-NONE count ({n_non_none}). "
                f"Cannot remove non-NONE entries; increase target-total."
            )
        desired_remove = n_total - target_total
        if desired_remove <= 0:
            print(f"[INFO] Dataset already at or below target total ({n_total} <= {target_total}). No removals.")
            desired_remove = 0
        remove_none = min(desired_remove, n_none)
        target_none = n_none - remove_none

    # Plan stratified keeping of NONEs by AR
    none_groups = group_none_by_ar(records)
    keep_plan, actually_kept_none = compute_removal_plan(none_groups, target_none, rng)

    # Sample indices to KEEP for each AR according to keep_plan
    keep_none_indices = []
    remove_none_indices = []

    for ar, idxs in none_groups.items():
        k_keep = keep_plan.get(ar, 0)
        if k_keep >= len(idxs):
            chosen_keep = list(idxs)
            chosen_remove = []
        else:
            chosen_keep = rng.sample(idxs, k_keep)
            chosen_remove = [i for i in idxs if i not in chosen_keep]
        keep_none_indices.extend(chosen_keep)
        remove_none_indices.extend(chosen_remove)

    # SAFETY: align to planned per-AR counts if any mismatch (shouldn't happen but we guard anyway)
    planned_total = sum(keep_plan.values())
    current_kept = len(keep_none_indices)
    if current_kept != planned_total:
        diff = planned_total - current_kept
        # Build per-AR current keep counts
        keep_counts = defaultdict(int)
        for i in keep_none_indices:
            ar = records[i].get("ar")
            keep_counts[ar] += 1

        if diff > 0:
            # Need to keep more: move from remove->keep, prioritizing ARs where we are below plan
            # Build AR->list(removed indices) for convenience
            removed_by_ar = defaultdict(list)
            for i in remove_none_indices:
                ar = records[i].get("ar")
                removed_by_ar[ar].append(i)
            # For each AR, add until reaching planned
            for ar, planned in keep_plan.items():
                while keep_counts.get(ar, 0) < planned and removed_by_ar[ar]:
                    i = removed_by_ar[ar].pop()
                    keep_none_indices.append(i)
                    keep_counts[ar] += 1
                    remove_none_indices.remove(i)
                    diff -= 1
                    if diff == 0:
                        break
                if diff == 0:
                    break
            # If still short (very unlikely), just pull arbitrarily from remaining removes
            if diff > 0 and remove_none_indices:
                extra = remove_none_indices[:diff]
                keep_none_indices.extend(extra)
                # keep_counts updated but not needed further
                remove_none_indices = remove_none_indices[diff:]
        elif diff < 0:
            # Need to remove some kept: move from keep->remove but DO NOT go below planned per-AR
            # Prefer ARs currently above their planned keep
            overfilled = [i for i in keep_none_indices if keep_counts[records[i].get("ar")] > keep_plan.get(records[i].get("ar"), 0)]
            rng.shuffle(overfilled)
            to_move = min(-diff, len(overfilled))
            moved = set(overfilled[:to_move])
            keep_none_indices = [i for i in keep_none_indices if i not in moved]
            remove_none_indices.extend(list(moved))

    # Compose output records: all non-NONE + kept NONE
    keep_set = set(non_none_indices) | set(keep_none_indices)
    out_records = [rec for i, rec in enumerate(records) if i in keep_set]
    removed_records = [rec for i, rec in enumerate(records) if i not in keep_set]

    # Summaries
    out_none = sum(1 for r in out_records if r.get("label") == LABEL_NONE)
    out_total = len(out_records)

    # Per-AR summary for NONEs (before/after) — CORRECTED
    def ar_counts_from_indices(records_list, indices):
        counts = defaultdict(int)
        for i in indices:
            ar = records_list[i].get("ar")
            counts[ar] += 1
        return counts

    def ar_counts_from_records(records_list):
        counts = defaultdict(int)
        for r in records_list:
            if r.get("label") == LABEL_NONE:
                ar = r.get("ar")
                counts[ar] += 1
        return counts

    before_counts = ar_counts_from_indices(records, none_indices)
    after_counts = ar_counts_from_records(out_records)

    print("=== Summary ===")
    print(f"Input: total={n_total}, NONE={n_none}, non-NONE={n_non_none}")
    print(f"Target: total={n_non_none + target_none}, NONE={target_none}")
    print(f"Output: total={out_total}, NONE={out_none}")
    print(f"Removed NONE: {len(remove_none_indices)}")
    print(f"AR groups: {len(before_counts)} (with NONEs)")
    print("Top 10 AR changes (AR: before -> after):")
    for ar in list(sorted(before_counts.keys(), key=lambda a: -before_counts[a]))[:10]:
        print(f"  AR {ar}: {before_counts[ar]} -> {after_counts.get(ar, 0)}")

    # Optional sanity checks
    num_groups_with_none = len(before_counts)
    if target_none >= num_groups_with_none:
        missing_ars = [ar for ar in before_counts if before_counts[ar] > 0 and after_counts.get(ar, 0) == 0]
        if missing_ars:
            print("[WARN] Some ARs lost all NONE entries despite target allowing ≥1 per AR:")
            print("       ", missing_ars[:20], ("... (+ more)" if len(missing_ars) > 20 else ""))

    # No file writes in dry-run
    if args.dry_run:
        print("[DRY-RUN] No files written.")
        return

    save_data(args.output, out_records)
    print(f"[OK] Wrote output dataset to: {args.output}")

    if args.removed:
        save_data(args.removed, removed_records)
        print(f"[OK] Wrote removed entries to: {args.removed}")


if __name__ == "__main__":
    main()
