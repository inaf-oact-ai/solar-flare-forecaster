#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

FILENAME_TIMESTAMP_RE = re.compile(r'(\d{8}_\d{6})')
FILENAME_AR_RE = re.compile(r'^(\d+)_')

def parse_args():
    ap = argparse.ArgumentParser(description="Build AR feature time series from CSV.")
    ap.add_argument("--input", required=True, help="Path to input CSV.")
    ap.add_argument("--output", required=True, help="Path to output JSON.")
    ap.add_argument("--base-cadence-min", type=int, default=12,
                    help="Cadence of raw images in minutes (default: 12).")
    ap.add_argument("--point-cadence-min", type=int, default=12,
                    help="Cadence between points in the output series (default: 12). "
                         "Must be a multiple of --base-cadence-min.")
    ap.add_argument("--series-length-points", type=int, default=120,
                    help="Number of points in each time series (default: 120; with 12-min cadence → 24h).")
    ap.add_argument("--moving-window-size", type=int, default=None,
                    help="Stride (in raw entries, i.e., base-cadence steps) between consecutive series starts. "
                         "Default: non-overlapping windows for the chosen cadence/length.")
    ap.add_argument("--forecast-horizon-h", type=int, default=24,
                    help="Forecasting horizon in hours (metadata only).")
    ap.add_argument("--expect-contiguous", action="store_true", default=True,
                    help="Enforce that timestamps are strictly contiguous at base cadence; break sequences at gaps.")
    return ap.parse_args()

def extract_ar_and_dt(fname: str) -> Tuple[int, datetime]:
    m_ar = FILENAME_AR_RE.search(fname)
    if not m_ar:
        raise ValueError(f"Cannot extract AR id from filename: {fname}")
    ar = int(m_ar.group(1))

    m_ts = FILENAME_TIMESTAMP_RE.search(fname)
    if not m_ts:
        raise ValueError(f"Cannot extract timestamp from filename: {fname}")
    ts_str = m_ts.group(1)
    dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    return ar, dt

def parse_row(row):
    if len(row) < 32:
        raise ValueError(f"Row has <32 fields: {row}")

    # take the LAST 32 fields to avoid off-by-one when extra cols exist
    row = [c.strip() for c in row[-32:]]

    feats = [float(row[i]) for i in range(29)]

    try:
        class_label = int(row[29])
    except Exception:
        class_label = None

    reg_label = normalize_reg_label(row[30])
    fname = row[31].strip()

    ar, dt = extract_ar_and_dt(fname)
    return {
        "ar": ar,
        "dt": dt,
        "features": feats,
        "class_label": class_label,
        "reg_label": reg_label,  # normalized
        "fname": fname,
    }

def normalize_reg_label(s: str) -> str:
    if s is None:
        return "0"
    s = str(s).strip().strip('"').strip("'").upper()
    if re.fullmatch(r'0+(\.0+)?', s):
        return "0"
    m = re.match(r'\s*([CMX])\s*([0-9]+(?:\.[0-9]+)?)', s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return s
    
def label_from_regression(reg_label: str):
    r = normalize_reg_label(reg_label)
    if r == "0" or r == "":
        return dict(flare_type="NONE", flare_id=0, label="NONE", id=0)
    t = r[0]
    if t == "C":
        return dict(flare_type="C", flare_id=1, label="NONE", id=0)
    if t == "M":
        return dict(flare_type="M", flare_id=2, label="M+", id=1)
    if t == "X":
        return dict(flare_type="X", flare_id=3, label="M+", id=1)
    return dict(flare_type="NONE", flare_id=0, label="NONE", id=0)

def contiguous_chunks(rows: List[Dict[str, Any]], base_cadence_min: int) -> List[List[Dict[str, Any]]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["dt"])
    chunks = [[rows[0]]]
    exp_delta = timedelta(minutes=base_cadence_min)
    for prev, cur in zip(rows, rows[1:]):
        if (cur["dt"] - prev["dt"]) == exp_delta:
            chunks[-1].append(cur)
        else:
            # break at the gap
            chunks.append([cur])
    return chunks

def build_series_from_chunk(chunk: List[Dict[str, Any]],
                            step_entries: int,
                            series_length_points: int,
                            point_cadence_min: int,
                            base_cadence_min: int,
                            moving_window_size: int) -> List[List[Dict[str, Any]]]:
    """
    From a contiguous chunk (at base cadence), build windows sampled every 'step_entries' raw entries.
    Each candidate window is validated so that successive timestamps differ by exactly point_cadence_min.
    """
    out = []
    needed = (series_length_points - 1) * step_entries + 1
    if moving_window_size <= 0:
        moving_window_size = 1

    exp_delta = timedelta(minutes=point_cadence_min)

    for start in range(0, len(chunk) - needed + 1, moving_window_size):
        idxs = [start + k * step_entries for k in range(series_length_points)]
        seq = [chunk[i] for i in idxs]

        # STRICT cadence check at point cadence
        ok = True
        for a, b in zip(seq, seq[1:]):
            if (b["dt"] - a["dt"]) != exp_delta:
                ok = False
                break
        if not ok:
            continue  # skip this series

        out.append(seq)
    return out

def main():
    args = parse_args()

    if args.point_cadence_min % args.base_cadence_min != 0:
        raise ValueError("--point-cadence-min must be a multiple of --base-cadence-min")

    step_entries = args.point_cadence_min // args.base_cadence_min
    print(f"step_entries: {step_entries}")

    if args.moving_window_size is None:
        # default to non-overlapping windows in raw-entry space
        args.moving_window_size = step_entries * args.series_length_points
        
    print(f"moving_window_size: {args.moving_window_size}")

    rows_by_ar = defaultdict(list)
    
    print(f"Reading input file {args.input} ...")
    with open(args.input, "r", newline="") as f:
        rdr = csv.reader(f)
        for raw in rdr:
            # naive header skip if present (allowing for variations)
            if raw and isinstance(raw[-1], str) and raw[-1].lower().endswith(".png"):
                pass
            try:
                rec = parse_row(raw)
            except Exception:
                raw2 = [c.strip() for c in raw]
                rec = parse_row(raw2)
            rows_by_ar[rec["ar"]].append(rec)

    print(f"#{len(rows_by_ar)} rows read ...")

    data_out = []


    
    for ar, rows in rows_by_ar.items():
        print(f"--> Creating chunks from ar {ar} ...")
        print(rows)
        
        if args.expect_contiguous:
            chunks = contiguous_chunks(rows, args.base_cadence_min)
        else:
            chunks = [sorted(rows, key=lambda r: r["dt"])]

        print(f"#{len(chunks)} chunks created for ar {ar} ...")

        chunk_counter= 0 
        for chunk in chunks:
            chunk_counter+= 1
            
            # Build cadence-validated series windows
            series_windows = build_series_from_chunk(
                chunk=chunk,
                step_entries=step_entries,
                series_length_points=args.series_length_points,
                point_cadence_min=args.point_cadence_min,
                base_cadence_min=args.base_cadence_min,
                moving_window_size=args.moving_window_size
            )
            
            print(f"Created #{len(series_windows)} series for chunk no. {chunk_counter}/{len(chunks)} for ar {ar} ...")

            for seq in series_windows:
                feat_series = {f"feat{i+1}": [] for i in range(29)}
                for item in seq:
                    for i, val in enumerate(item["features"]):
                        feat_series[f"feat{i+1}"].append(val)

                first_dt = seq[0]["dt"]
                last_dt = seq[-1]["dt"]
                lbl_meta = label_from_regression(seq[-1]["reg_label"])

                record = {
                    "ar": ar,
                    "npoints": args.series_length_points,
                    "dt": args.point_cadence_min,
                    "forecast_horizon": f"{args.forecast_horizon_h}h",
                    "date_start": first_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "date_end": last_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    **lbl_meta,
                    **feat_series,
                }
                data_out.append(record)

    print(f"Saving data to file {args.output} ...")
    with open(args.output, "w") as f:
        json.dump({"data": data_out}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data_out)} series from {len(rows_by_ar)} ARs to {args.output}")
    if data_out:
        print("First series summary:",
              {"ar": data_out[0]["ar"], "npoints": data_out[0]["npoints"],
               "dt": data_out[0]["dt"], "date_start": data_out[0]["date_start"],
               "date_end": data_out[0]["date_end"], "label": data_out[0]["label"]})

if __name__ == "__main__":
    main()

