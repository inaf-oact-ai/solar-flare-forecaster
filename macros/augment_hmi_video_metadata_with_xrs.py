#!/usr/bin/env python3
"""
Augment HMI video metadata with GOES XRS ratio & flare history + TS-only labels,
with strict time-length consistency checks and label agreement stats.

Adds per item:
  - "timestamps": [ISO...]
  - "xrs_flux_ratio": [float...]
  - "flare_hist": [float...]                (optional)
  - "xrs_satellite": "goes16"               (example)
  - "xrs_band": "long" | "short"
  - "xrs_resample": "1min"
  - "n_video_frames": int
  - "n_minutes": int                        (= len(xrs_flux_ratio))
  - "flare_ts_type": "NONE"|"C"|"M"|"X"
  - "flare_ts_id": 0|1|2|3

At end, prints per-satellite:
  - total processed, equal labels, different labels, missing/unknown labels

Usage (example):
  python augment_hmi_metadata_with_xrs.py \
    --hmi-json /path/to/hmi_meta.json \
    --out-json /path/to/hmi_meta_augmented.json \
    --science-root /home/riggi/Data/SolarData/GOES/NetCDF/1minAvg \
    --background-root /home/riggi/Data/SolarData/GOES/NetCDF/dailyBkg \
    --events-file /home/riggi/Data/SolarData/FlareEvents/eventList.csv \
    --satellite 16 \
    --band long \
    --resample 1min \
    --min-coverage 0.95 \
    --emit-history-channel \
    --history-encoding binary \
    --skip-ABflares-in-history \
    --label-window-hours 24 \
    --forecast-gap-hours 0.0
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import csv
import os

# == Reuse your existing helpers from your preprocessing module ==
# (Same file you shared previously — keeps semantics aligned.)
from make_xrs_ts_data import (   # :contentReference[oaicite:1]{index=1}
    open_science_files,
    read_background_dir,
    load_events,
    discover_time_name,
    discover_science_var,
    build_history_channel,
)

# --- Label helpers (mirroring your existing logic) ---
LABEL_REMAP = {"NONE":"NONE","A":"NONE","B":"NONE","C":"C","M":"M","X":"X"}
TS_ID_MAP   = {"NONE":0, "C":1, "M":2, "X":3}

def choose_label(classes: List[str]) -> str:
    """Return highest-severity label among classes; includes A/B if present."""
    if "X" in classes: return "X"
    if "M" in classes: return "M"
    if "C" in classes: return "C"
    if "B" in classes: return "B"
    if "A" in classes: return "A"
    return "NONE"

def remap_label(label: str) -> str:
    return LABEL_REMAP.get(label, "NONE")


def parse_args():
    p = argparse.ArgumentParser(description="Augment HMI metadata with GOES XRS and TS-only labels (with consistency checks)")
    p.add_argument("--hmi-json", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--science-root", type=Path, required=True)
    p.add_argument("--background-root", type=Path, required=True)
    p.add_argument("--events-file", type=Path, required=True)
    p.add_argument("--satellite", type=str, default="16")
    p.add_argument("--science-file-glob", type=str, default="*.nc")
    p.add_argument("--background-file-glob", type=str, default="*.nc")
    p.add_argument("--band", choices=["long","short"], default="long")
    p.add_argument("--science-var", default=None)
    p.add_argument("--background-var", default=None)
    p.add_argument("--time-var", default=None)
    p.add_argument("--resample", default="1min")
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--emit-history-channel", action="store_true")
    p.add_argument("--history-encoding", choices=["binary","ordinal"], default="binary")
    p.add_argument("--skip-ABflares-in-history", action="store_true")
    p.add_argument("--include-timestamps", dest="include_timestamps", action="store_true")
    p.add_argument("--no-timestamps", dest="include_timestamps", action="store_false")
    p.set_defaults(include_timestamps=True)
    p.add_argument("--only-matched", action="store_true", help="If set, drop items where XRS augmentation failed (xrs_flux_ratio is None).")
    

    # TS-only label window configuration
    p.add_argument("--label-window-hours", type=float, default=24.0,
                   help="Length of the forward-looking window used to compute flare_ts_type after t_end.")
    p.add_argument("--forecast-gap-hours", type=float, default=0.0,
                   help="Gap Δ hours between t_end and the start of the label window.")
    return p.parse_args()


def load_science_and_background(args) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    sat_name = f"goes{args.satellite}"
    sci_dir  = args.science_root / sat_name
    bg_dir   = args.background_root / sat_name
    if not sci_dir.exists():
        raise FileNotFoundError(f"Science dir not found: {sci_dir}")
    if not bg_dir.exists():
        raise FileNotFoundError(f"Background dir not found: {bg_dir}")

    ds = open_science_files(sci_dir, args.science_file_glob)
    tname = args.time_var or discover_time_name(ds)
    vname = args.science_var or discover_science_var(ds, args.band)

    # science → DataFrame(time, science), resampled to cadence
    sci = (
        ds[[vname]].to_dataframe().reset_index()
        .rename(columns={tname: "timestamp", vname: "science"})
    )
    sci["timestamp"] = pd.to_datetime(sci["timestamp"], utc=True)
    sci = (
        sci.set_index("timestamp")
           .sort_index()
           .resample(args.resample).mean()
           .astype({"science": "float64"})
    )

    # background (daily)
    bg = read_background_dir(bg_dir, args.background_file_glob, args.band, args.background_var)
    bg["date"] = bg["timestamp"].dt.floor("D")
    return sci, bg, sat_name


def slice_ratio_on_window(start: pd.Timestamp,
                          end: pd.Timestamp,
                          sci: pd.DataFrame,
                          bg: pd.DataFrame,
                          min_coverage: float) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    start = pd.to_datetime(start, utc=True)
    end   = pd.to_datetime(end,   utc=True)
    if not (end > start):
        raise ValueError(f"Invalid window: start {start} >= end {end}")

    # 1-minute grid [start, end)
    grid = pd.date_range(start=start, end=end, freq="1min", inclusive="left", tz="UTC")
    if grid.empty:
        raise ValueError(f"Empty grid for window [{start},{end})")

    seg = sci.reindex(grid)
    frac_valid = seg["science"].notna().mean()
    if np.isnan(frac_valid) or frac_valid < min_coverage:
        raise RuntimeError(f"Coverage {frac_valid:.3f} below threshold {min_coverage:.3f}")

    seg["science"] = seg["science"].interpolate(limit_direction="both")

    # join background per-day
    day_df = pd.DataFrame(index=grid)
    day_df["date"] = day_df.index.floor("D")
    bg_day = bg[["date", "background"]].drop_duplicates("date").set_index("date")
    day_df = day_df.join(bg_day, on="date")

    if (day_df["background"] <= 0).any() or not np.isfinite(day_df["background"]).all():
        raise RuntimeError("Invalid background (<=0 or non-finite) encountered in the window")

    ratio = (seg["science"].values.astype("float64") / day_df["background"].values.astype("float64")).astype("float32")
    return grid, ratio


def compute_ts_only_label(t_end: pd.Timestamp,
                          events_df: pd.DataFrame,
                          label_window_h: float,
                          gap_h: float) -> str:
    """Label based on highest class in (t_end+gap, t_end+gap+label_window]."""
    gap     = pd.Timedelta(hours=gap_h)
    looklen = pd.Timedelta(hours=label_window_h)
    label_start = pd.to_datetime(t_end, utc=True) + gap
    label_end   = label_start + looklen
    occ = events_df[(events_df["t_occ"] > label_start) & (events_df["t_occ"] <= label_end)]
    label_raw = choose_label(occ["class"].tolist())
    return remap_label(label_raw)   # map A/B → NONE


def main():
    args = parse_args()
    
    # Init CSV writer for spot-check
    csv_path = args.out_json.with_suffix(".spotcheck.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow([
        "id","ar",
        "fname_start", "fname_end",
        "t_start","t_end",
        "video_label","ts_label"
#        "video_label","ts_label","ts_id","equal_flag"
    ])

    # Load input metadata
    meta_path: Path = args.hmi_json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert isinstance(meta, dict) and isinstance(meta.get("data"), list), "Input JSON must have top-level 'data' list"

    # Load GOES assets (single satellite run)
    print("Reading GOES events data ...")
    events_df = load_events(args.events_file)  # keeps start/end for history channel
    
    print("Reading science & bkg data ...")
    sci, bg, sat_name = load_science_and_background(args)

    # Stats (per satellite)
    n_total = 0
    n_ok = 0
    n_skipped = 0
    n_equal = 0
    n_diff = 0
    n_missing = 0

    out_items: List[Dict[str, Any]] = []
    
    ntot_entries= len(meta["data"])
    
    print(f"Start looping over {ntot_entries} entries ...")

    for item in meta["data"]:
        n_total += 1
        out = dict(item)
        
        if n_total%1000==0:
            print(f"--> Processed {n_total}/{ntot_entries} ...")

        try:
            t_start = pd.to_datetime(item["t_start"], utc=True)
            t_end   = pd.to_datetime(item["t_end"],   utc=True)
            if not (t_end > t_start):
                raise ValueError("t_end must be > t_start")

            # Build per-minute ratio
            grid, ratio = slice_ratio_on_window(t_start, t_end, sci, bg, args.min_coverage)

            # Optional history
            hist = None
            if args.emit_history_channel:
                hist = build_history_channel(
                    index=grid,
                    args=args,
                    events_df=events_df,
                    encoding=args.history_encoding,
                ).values.astype("float32")

            # === Consistency checks ===
            expected_minutes = int((t_end - t_start) / pd.Timedelta(minutes=1))
            if len(ratio) != expected_minutes:
                raise RuntimeError(f"Length mismatch: ratio={len(ratio)} vs expected_minutes={expected_minutes}")
            if hist is not None and len(hist) != expected_minutes:
                raise RuntimeError(f"Length mismatch: hist={len(hist)} vs expected_minutes={expected_minutes}")

            # Lightweight video info (frame count)
            filepaths = item.get("filepaths", []) or []
            n_frames = int(len(filepaths))

            # Attach outputs
            if args.include_timestamps:
                out["timestamps"] = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in grid]
            out["xrs_flux_ratio"] = ratio.tolist()
            if hist is not None:
                out["flare_hist"] = hist.tolist()

            out["xrs_satellite"] = sat_name
            out["xrs_band"] = args.band
            out["xrs_resample"] = args.resample
            out["n_video_frames"] = n_frames
            out["n_minutes"] = int(len(ratio))

            # === Time-series-only label & id ===
            ts_type = compute_ts_only_label(t_end, events_df, args.label_window_hours, args.forecast_gap_hours)
            out["flare_ts_type"] = ts_type
            out["flare_ts_id"]   = int(TS_ID_MAP.get(ts_type, 0))

            # === Compare with video label (if present) ===
            equal_flag = 0
            video_type = out.get("flare_type", None)
            if isinstance(video_type, str) and len(video_type) > 0:
                video_type_letter = video_type.strip().upper()[:1]  # "C+" → "C"
                if video_type_letter not in {"N","C","M","X"}:
                    # Map "A"/"B" to NONE, or unknown to missing
                    if video_type_letter in {"A","B"}:
                        video_type_letter = "NONE"
                    elif video_type_letter == "0":  # in case "NONE" got encoded as "0" elsewhere
                        video_type_letter = "NONE"
                    else:
                        n_missing += 1
                        video_type_letter = None
                if video_type_letter is None:
                    pass  # counted as missing
                else:
                    if video_type_letter == ts_type:
                        n_equal += 1
                    else:
                        n_diff  += 1
            else:
                n_missing += 1

            n_ok += 1
            out_items.append(out)
            
            
            # Write to CSV
            csv_writer.writerow([
                out.get("id",""),
                out.get("ar",""),
                os.path.basename(filepaths[0]),
                os.path.basename(filepaths[len(filepaths)-1]),
                out.get("t_start",""),
                out.get("t_end",""),
                out.get("flare_type",""),
                ts_type,
                #TS_ID_MAP.get(ts_type,0),
                #equal_flag
            ])

        except Exception as e:
            out["xrs_flux_ratio"] = None
            if args.emit_history_channel:
                out["flare_hist"] = None
            out["xrs_error"] = str(e)
            out["xrs_satellite"] = sat_name
            n_skipped += 1
            
            if not args.only_matched:
                out_items.append(out)

    # Wrap + stats block
    out_json = {
        "data": out_items,
        "stats": {
            "satellite": sat_name,
            "processed": n_total,
            "ok": n_ok,
            "skipped": n_skipped,
            "label_agreement": {
                "equal": n_equal,
                "different": n_diff,
                "missing_or_unknown_video_label": n_missing
            }
        }
    }

    # Save
    out_path: Path = args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    # Print per-satellite summary
    print("======== XRS Augmentation Summary ========")
    print(f"Satellite: {sat_name}")
    print(f"Processed: {n_total} | OK: {n_ok} | Skipped: {n_skipped}")
    print(f"Label agreement (video vs TS): equal={n_equal}, different={n_diff}, missing/unknown={n_missing}")
    print("==========================================")

if __name__ == "__main__":
    main()

