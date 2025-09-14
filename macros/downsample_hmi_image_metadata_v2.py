#!/usr/bin/env python3
"""
downsample_hmi_image_metadata.py

Downsample selected flare classes in an HMI image JSON dataset, stratified by AR (active region).

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
  --target-total <N>           : desired total number of entries in the output
or one/both of:
  --target-none  <N>           : desired number of "NONE" entries in the output
  --target-c     <N>           : desired number of "C"    entries in the output

Only the targeted labels (NONE and/or C) are removed. Other labels are always kept.

Removal is stratified by "ar" so that small-AR groups are not fully eliminated unless mathematically
unavoidable (e.g., you need fewer entries than the number of AR groups).

Examples:
  python downsample_hmi_image_metadata.py --input in.json --output out.json --target-total 50000 --seed 42
  python downsample_hmi_image_metadata.py --input in.json --output out.json --target-none 12000 --removed removed.json
  python downsample_hmi_image_metadata.py --input in.json --output out.json --target-none 12000 --target-c 8000
"""

import argparse
import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

LABEL_NONE = "NONE"
LABEL_C    = "C"


def parse_args():
    p = argparse.ArgumentParser(description="Downsample selected classes (NONE and/or C) stratified by AR.")
    p.add_argument("--input", required=True, help="Path to input JSON file")
    p.add_argument("--output", required=True, help="Path to output JSON file")

    # Either total target OR per-label targets
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--target-total", type=int, help="Desired total number of entries in output")
    grp.add_argument("--per-label", action="store_true",
                     help="Enable per-label targeting via --target-none / --target-c")

    # Per-label targets (used only if --per-label is set)
    p.add_argument("--target-none", type=int, default=None, help="Desired number of NONE entries in the output")
    p.add_argument("--target-c",    type=int, default=None, help="Desired number of C entries in the output")

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


def group_by_ar(records: List[dict], label: str) -> Dict[int, List[int]]:
    """
    Returns dict: ar -> list of indices (positions in 'records') for entries with given label in that AR.
    """
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        if rec.get("label") == label:
            ar = rec.get("ar")
            groups[ar].append(idx)
    return groups


def compute_removal_plan(
    groups: Dict[int, List[int]],
    keep_total: int,
    rng: random.Random
) -> Tuple[Dict[int, int], int]:
    """
    Given label groups (ar -> indices), and the desired total to keep for that label,
    compute how many to keep per AR (then removals are implied).
    Strategy:
      1) If keep_total >= #groups: guarantee at least 1 kept per group (when possible),
         then distribute remaining proportionally to group sizes (via randomized rounding).
      2) If keep_total < #groups: choose keep_total groups to keep (1 each),
         sampled with probability proportional to group size; other groups keep 0.
    Returns: (keep_per_ar, actually_kept)
    """
    ars = list(groups.keys())
    sizes = {ar: len(groups[ar]) for ar in ars}
    G = len(ars)
    total = sum(sizes.values())

    # Clamp the desired total to feasible range
    keep_total = max(0, min(keep_total, total))
    keep_per_ar = {ar: 0 for ar in ars}

    if keep_total == 0 or G == 0:
        return keep_per_ar, 0

    if keep_total >= G:
        # Start by giving one to each non-empty group
        base_allocate = sum(1 for ar in ars if sizes[ar] > 0)
        for ar in ars:
            if sizes[ar] > 0:
                keep_per_ar[ar] = 1

        remaining = keep_total - base_allocate
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
        # keep_total < number of groups
        probs = [sizes[ar] for ar in ars]
        chosen_groups = set(weighted_sample_without_replacement(ars, probs, keep_total, rng))
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
    idx_none = [i for i, r in enumerate(records) if r.get("label") == LABEL_NONE]
    idx_c    = [i for i, r in enumerate(records) if r.get("label") == LABEL_C]
    idx_other = [i for i, r in enumerate(records) if r.get("label") not in (LABEL_NONE, LABEL_C)]

    n_total = len(records)
    n_none  = len(idx_none)
    n_c     = len(idx_c)
    n_other = len(idx_other)

    # Resolve targets
    if args.target_total is not None:
        target_total = args.target_total
        if target_total < n_other:
            raise ValueError(
                f"--target-total={target_total} is smaller than non-removable count (others={n_other}). "
                f"Only NONE/C can be removed; increase target-total."
            )

        desired_remove = n_total - target_total
        if desired_remove <= 0:
            print(f"[INFO] Dataset already at or below target total ({n_total} <= {target_total}). No removals.")
            desired_remove = 0

        # Remove from NONE first, then from C if still needed
        remove_none = min(desired_remove, n_none)
        remaining = desired_remove - remove_none
        remove_c = min(max(0, remaining), n_c)

        target_none = n_none - remove_none
        target_c    = n_c - remove_c

    else:
        # Per-label targets
        if not args.per_label:
            raise ValueError("Internal error: choose either --target-total or --per-label")
        if args.target_none is None and args.target_c is None:
            raise ValueError("When using --per-label, specify at least one of --target-none or --target-c.")

        # If not specified, keep all current entries for that label
        target_none = n_none if args.target_none is None else max(0, min(args.target_none, n_none))
        target_c    = n_c    if args.target_c    is None else max(0, min(args.target_c,    n_c))

        if args.target_none is not None and args.target_none > n_none:
            print(f"[INFO] Requested --target-none={args.target_none} > current NONE={n_none}. No removals for NONE.")
        if args.target_c is not None and args.target_c > n_c:
            print(f"[INFO] Requested --target-c={args.target_c} > current C={n_c}. No removals for C.")

        target_total = n_other + target_none + target_c

    # Plans per label (stratified by AR)
    keep_none_plan = {}
    keep_c_plan = {}
    actually_kept_none = 0
    actually_kept_c = 0

    if target_none != n_none:
        none_groups = group_by_ar(records, LABEL_NONE)
        keep_none_plan, actually_kept_none = compute_removal_plan(none_groups, target_none, rng)
    else:
        actually_kept_none = n_none

    if target_c != n_c:
        c_groups = group_by_ar(records, LABEL_C)
        keep_c_plan, actually_kept_c = compute_removal_plan(c_groups, target_c, rng)
    else:
        actually_kept_c = n_c

    # Sample indices to KEEP for each label according to their plans
    keep_none_indices, remove_none_indices = _apply_keep_plan(records, LABEL_NONE, keep_none_plan, rng)
    keep_c_indices,    remove_c_indices    = _apply_keep_plan(records, LABEL_C,    keep_c_plan,    rng)

    # Compose output records: all 'other' + kept NONE + kept C
    keep_set = set(idx_other) | set(keep_none_indices) | set(keep_c_indices)
    out_records = [rec for i, rec in enumerate(records) if i in keep_set]
    removed_records = [rec for i, rec in enumerate(records) if i not in keep_set]

    # Summaries
    out_none = sum(1 for r in out_records if r.get("label") == LABEL_NONE)
    out_c    = sum(1 for r in out_records if r.get("label") == LABEL_C)
    out_total = len(out_records)

    # Per-AR summaries
    before_none_counts = _counts_from_indices(records, idx_none)
    after_none_counts  = _counts_from_records(out_records, LABEL_NONE)

    before_c_counts = _counts_from_indices(records, idx_c)
    after_c_counts  = _counts_from_records(out_records, LABEL_C)

    print("=== Summary ===")
    print(f"Input: total={n_total}, NONE={n_none}, C={n_c}, others={n_other}")
    print(f"Target: total={target_total}, NONE={target_none}, C={target_c}, others={n_other}")
    print(f"Output: total={out_total}, NONE={out_none}, C={out_c}, others={n_other}")
    print(f"Removed NONE: {len(remove_none_indices)} | Removed C: {len(remove_c_indices)}")

    print(f"AR groups with NONE: {len(before_none_counts)} | with C: {len(before_c_counts)}")

    def _print_top(label, before, after, topk=10):
        print(f"Top {topk} AR changes for {label} (AR: before -> after):")
        for ar in list(sorted(before.keys(), key=lambda a: -before[a]))[:topk]:
            print(f"  AR {ar}: {before[ar]} -> {after.get(ar, 0)}")

    _print_top("NONE", before_none_counts, after_none_counts)
    _print_top("C",    before_c_counts,    after_c_counts)

    # Sanity: if we asked to keep ≥ number of ARs, warn if any AR lost all entries for that label
    _warn_if_missing_ars("NONE", target_none, before_none_counts, after_none_counts)
    _warn_if_missing_ars("C",    target_c,    before_c_counts,    after_c_counts)

    # No file writes in dry-run
    if args.dry_run:
        print("[DRY-RUN] No files written.")
        return

    save_data(args.output, out_records)
    print(f"[OK] Wrote output dataset to: {args.output}")

    if args.removed:
        save_data(args.removed, removed_records)
        print(f"[OK] Wrote removed entries to: {args.removed}")


def _apply_keep_plan(records: List[dict], label: str, keep_plan: Dict[int, int], rng: random.Random):
    """Return (keep_indices, remove_indices) for a given label based on the per-AR keep plan."""
    if not keep_plan:
        # Plan not needed (keeping everything as-is)
        indices = [i for i, r in enumerate(records) if r.get("label") == label]
        return indices, []

    groups = group_by_ar(records, label)
    keep_indices = []
    remove_indices = []

    for ar, idxs in groups.items():
        k_keep = keep_plan.get(ar, 0)
        if k_keep >= len(idxs):
            chosen_keep = list(idxs)
            chosen_remove = []
        else:
            chosen_keep = rng.sample(idxs, k_keep)
            chosen_remove = [i for i in idxs if i not in chosen_keep]
        keep_indices.extend(chosen_keep)
        remove_indices.extend(chosen_remove)

    # Safety alignment with planned counts (rare)
    planned_total = sum(keep_plan.values())
    if len(keep_indices) != planned_total:
        diff = planned_total - len(keep_indices)
        keep_counts = defaultdict(int)
        for i in keep_indices:
            keep_counts[records[i].get("ar")] += 1

        if diff > 0:
            removed_by_ar = defaultdict(list)
            for i in remove_indices:
                removed_by_ar[records[i].get("ar")].append(i)
            for ar, planned in keep_plan.items():
                while keep_counts.get(ar, 0) < planned and removed_by_ar[ar]:
                    i = removed_by_ar[ar].pop()
                    keep_indices.append(i)
                    keep_counts[ar] += 1
                    remove_indices.remove(i)
                    diff -= 1
                    if diff == 0:
                        break
                if diff == 0:
                    break
            if diff > 0 and remove_indices:
                extra = remove_indices[:diff]
                keep_indices.extend(extra)
                remove_indices = remove_indices[diff:]
        elif diff < 0:
            overfilled = [i for i in keep_indices if keep_counts[records[i].get("ar")] > keep_plan.get(records[i].get("ar"), 0)]
            rng.shuffle(overfilled)
            to_move = min(-diff, len(overfilled))
            moved = set(overfilled[:to_move])
            keep_indices = [i for i in keep_indices if i not in moved]
            remove_indices.extend(list(moved))

    return keep_indices, remove_indices


def _counts_from_indices(records_list, indices):
    counts = defaultdict(int)
    for i in indices:
        ar = records_list[i].get("ar")
        counts[ar] += 1
    return counts


def _counts_from_records(records_list, label):
    counts = defaultdict(int)
    for r in records_list:
        if r.get("label") == label:
            counts[r.get("ar")] += 1
    return counts


def _warn_if_missing_ars(label, target_keep, before_counts, after_counts):
    num_groups_with_label = len(before_counts)
    if target_keep >= num_groups_with_label:
        missing_ars = [ar for ar in before_counts if before_counts[ar] > 0 and after_counts.get(ar, 0) == 0]
        if missing_ars:
            print(f"[WARN] Some ARs lost all {label} entries despite target allowing ≥1 per AR:")
            print("       ", missing_ars[:20], ("... (+ more)" if len(missing_ars) > 20 else ""))


if __name__ == "__main__":
    main()

