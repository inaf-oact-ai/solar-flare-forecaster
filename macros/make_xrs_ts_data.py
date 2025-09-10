#!/usr/bin/env python3
"""
GOES XRS Science/Background Preprocessor (v2 - flat CSV support)
----------------------------------------------------------------
Builds aligned fixed-length time series per satellite based on
daily background entries, computes science/background ratio, and assigns
a label in {"NONE","C","M","X"} depending on whether any flare of that
class occurs in the *next* window_hours after the series end.

Optionally writes a *flat* CSV where each row is:
t0, x1, x2, ..., xN, label

Usage example
-------------
python prep_goes_series_v2.py \
  --science-root /path/to/science \
  --background-root /path/to/background \
  --events-file /path/to/eventList.txt \
  --satellites 08 09 10 11 12 13 14 15 16 17 18 \
  --window-hours 24 \
  --band long \
  --outdir /path/to/out \
  --emit-flat-csv \
  --flat-csv-path /path/to/out/series_flat.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import re
import json

COMMON_TIME_NAMES = ["time", "timestamp", "datetime", "date_time"]
SCIENCE_NAME_CANDIDATES = {
    "long":  ["xrsb_flux", "long_flux", "xrs_long", "goes_xrs_long", "flux_long"],
    "short": ["xrsa_flux", "short_flux", "xrs_short", "goes_xrs_short", "flux_short"],
}
#SEVERITY = {"NONE": 0, "C": 1, "M": 2, "X": 3}
SEVERITY = {"NONE": 0, "A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
LABEL_REMAP= {"NONE": "NONE", "A": "NONE", "B": "NONE", "C": "C", "M": "M", "X": "X"}
LABEL2ID= {"NONE": 0, "C": 1, "M": 2, "X": 3}

##############################
##   ARGUMENTS
##############################
def parse_args():
	""" Script arguments """
    
	p = argparse.ArgumentParser(description="GOES XRS science/background series builder & labeler (flat CSV enabled)")
	p.add_argument("--science-root", required=True, type=Path)
	p.add_argument("--background-root", required=True, type=Path)
	p.add_argument("--events-file", required=True, type=Path)
	p.add_argument("--satellites", nargs="+", default=["08","09","10","11","12","13","14","15","16","17","18"])
	p.add_argument("--window-hours", type=int, default=24)
	p.add_argument("--band", choices=["long","short"], default="long")
	p.add_argument("--science-var", default=None)
	p.add_argument("--background-var", default=None)
	p.add_argument("--time-var", default=None)
	p.add_argument("--min-coverage", type=float, default=0.95, help="Required fraction of non‑NaNs")
	p.add_argument("--outdir", required=True, type=Path)
	p.add_argument("--resample", default="1min", help="Resampling cadence of science flux data")
	p.add_argument("--science-file-glob", default="*.nc")
	p.add_argument("--background-file-glob", default="*.nc")
	p.add_argument("--dry-run", action="store_true")
	p.add_argument("--emit-flat-csv", action="store_true")
	p.add_argument("--flat-csv-path", type=Path, default=None)
	p.add_argument("--outfile-json", type=Path, default=None)
	p.add_argument(
		"--skip-inwindow-atleast",
		choices=["NONE","A","B","C","M","X"],
		default="NONE",
		help="If set, skip any series that contains a flare within the input window with class >= the given threshold (e.g., C drops C/M/X; M drops M/X)."
	)
	p.add_argument(
		"--skip-date-ranges",
		type=str,
		nargs="+",
		default=[],
		help="List of date ranges (YYYY-MM:YYYY-MM) to skip entirely based on series start date. Example: --skip-date-ranges 1995-01:1995-12 2003-06:2004-02"
	)
	p.add_argument(
		"--skip-years",
		type=int,
		nargs="+",
		default=[],
		help="List of years to skip entirely (based on the series start day from background). Example: --skip-years 1995 1998"
	)
	p.add_argument(
		"--forecast-gap-hours",
		type=float,
		default=0.0,
		help="Gap Δ (hours) between the input window end and the start of the labeling window. Label window becomes (end+Δ, end+Δ+window]. Default: 0 (no gap)."
	)
	p.add_argument(
		"--emit-history-channel",
		action="store_true",
		help="If set, add a historical flare channel H(t) on the 1-min grid: 0 outside flares, >0 during [t_start, t_end]."
	)
	p.add_argument(
		"--history-encoding",
		choices=["binary", "ordinal"],
		default="binary",
		help="How to encode H(t): 'binary' = 1 during any flare; 'ordinal' = A=1, B=2, C=3, M=4, X=5 (max if overlaps)."
	)
	p.add_argument(
                "--skip-ABflares-in-history",
                action="store_true",
                help="If set, do not consider A & B flare types in history."
        )

	return p.parse_args()

def discover_time_name(ds: xr.Dataset):
	""" Auto‑detects variable names in a loaded xarray Dataset. Looks in ds, ds.coords, then any datetime‑like var """
	for n in COMMON_TIME_NAMES:
		if n in ds or n in ds.coords:
			return n
			
	for c in ds.coords:
		if np.issubdtype(ds.coords[c].dtype, np.datetime64):
			return c
    
	for v in ds.data_vars:
		if np.issubdtype(ds[v].dtype, np.datetime64):
			return v

	raise KeyError("Could not detect time coordinate")

def discover_science_var(ds: xr.Dataset, band: str):
	""" Auto‑detects variable names in science flux dataset. Tries common names (e.g., xrsb_flux for long band), then a regex like xrsb/xrsa """
	
	for cand in SCIENCE_NAME_CANDIDATES[band]:
		if cand in ds:
			return cand
    
	lower_map = {v.lower(): v for v in ds.data_vars}
	for cand in SCIENCE_NAME_CANDIDATES[band]:
		if cand.lower() in lower_map:
			return lower_map[cand.lower()]
    
	pattern = "xrs" + ("b" if band=="long" else "a")
	for v in ds.data_vars:
		if re.search(pattern, v, re.IGNORECASE):
			return v
			
	raise KeyError("Science var not found")

def open_science_files(science_dir: Path, file_glob: str):
	""" 
		Open science flux data 
			- Finds all NetCDFs for a satellite and opens them as a single dataset via xarray.open_mfdataset.
			- Tries combine='by_coords' first; falls back to concat_dim='time' if needed.
	"""
    
	files = sorted(science_dir.glob(file_glob))
	if not files:
		raise FileNotFoundError(f"No science files in {science_dir}")
    
	try:
		ds = xr.open_mfdataset(files, combine='by_coords', engine="netcdf4")
	except Exception:
		ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', engine="netcdf4")
    
	return ds

def read_background_dir(bg_dir: Path, file_glob: str, band: str, background_var: str | None):
	""" 
		Read background data 
			- Finds a timestamp column (timestamp/date/time/datetime) and a background column (or var) using either your --background-var or common names like background_long, background, bkg.
			- Collapses to one entry per day (keeps the first per day), returning a DataFrame with timestamp (daily reference) and background
	"""

	# - Check for files    
	files = sorted(bg_dir.glob(file_glob))
	if not files:
		raise FileNotFoundError(f"No background files in {bg_dir}")
    
	# - Read bkg data
	dfs = []
	for f in files:
		ds = xr.open_dataset(f, engine="netcdf4")
		time_name = discover_time_name(ds)
		
		if background_var and background_var in ds:
			bg_name = background_var
		else:
			cand_names = [f"background_{band}", "background", f"bkg_{band}", "bkg"]
			bg_name = next((c for c in cand_names if c in ds), None)
			if not bg_name:
				raise KeyError(f"No background var in {f}")
			
		df = ds[[bg_name]].to_dataframe().reset_index()
		df = df.rename(columns={time_name: "timestamp", bg_name: "background"})
		df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
		dfs.append(df)
            
	all_bg = pd.concat(dfs, ignore_index=True).dropna(subset=["timestamp"]).sort_values("timestamp")
	all_bg["date"] = all_bg["timestamp"].dt.floor("D")
	all_bg = all_bg.groupby("date", as_index=False).first()[["timestamp", "background"]]
    
	return all_bg

def load_events(events_file: Path):
	""" 
		Read flare event data 
			- Normalizes column names to read timestamp_start (or timestamp if that’s missing).
			- Extracts the flare class letter from strings like C1.2, M3.0, X2.1.
			- Keeps only rows with classes in {C,M,X} and returns a tidy DataFrame with:
					t_occ: occurrence time (uses peak time when present),
					class: C, M, or X
	"""

	# - Read CSV flare list
	df = pd.read_csv(events_file)
	colmap = {
		"timestamp": "timestamp",
		"timestamp_start": "timestamp_start",
		"start_time": "timestamp_start",
		"timestamp_end": "timestamp_end",
		"flare_type": "flare_type",
		"class": "flare_type"
	}
	
	# - Rename data columns
	ren = {c: colmap[c.lower()] for c in df.columns if c.lower() in colmap}
	df = df.rename(columns=ren)

	for tcol in ["timestamp_start", "timestamp", "timestamp_end"]:
		if tcol in df.columns:
			df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
      
	def class_letter(s):
		if isinstance(s,str) and s:
			m = re.match(r"\s*([A,B,C,M,X])", s, re.I)
			return m.group(1).upper() if m else None
		return None
		
	
	df["class"] = df["flare_type"].apply(class_letter) if "flare_type" in df.columns else None
	df["sev"] = df["class"].map(SEVERITY)
	
	# - Use peak if present; fall back to start
	###df["t_occ"] = df["timestamp_start"] if "timestamp_start" in df.columns else df["timestamp"]
	#df["t_occ"] = df["timestamp"]
	df["t_occ"] = df["timestamp"].where(df["timestamp"].notna(), df["timestamp_start"])
	df = df.dropna(subset=["t_occ"])
	df = df[df["class"].isin(["A","B","C","M","X"])]
	
	###df = df[df["class"].isin(["C","M","X"])][["t_occ","class"]].sort_values("t_occ")
	#df = df[df["class"].isin(["A","B","C","M","X"])][["t_occ","class"]].sort_values("t_occ")
    
	# IMPORTANT: keep start/end so history channel can be built
	return (
		df[["t_occ", "class", "sev", "timestamp_start", "timestamp_end", "flare_type"]]
		.sort_values("t_occ")
		.reset_index(drop=True)
	) 
    
	#return df
	


def choose_label(classes):
	""" Given a list of classes inside a look‑ahead window, returns the highest severity """
	if "X" in classes: return "X"
	if "M" in classes: return "M"
	if "C" in classes: return "C"
	if "B" in classes: return "B"
	if "A" in classes: return "A"
	return "NONE"
	
def remap_label(label):
	""" Remap label """

	if label in LABEL_REMAP:
		return LABEL_REMAP[label]
		
	print(f"WARN: Unexpected label given ({label}), returning NONE!")
	return "NONE"


#class FlatCSVWriter:
#	""" 
#		Class to write flare time series data to CSV 
#			- Writes header: t0,x1,…,xN,label.
#			- Enforces fixed length N (derived from window_hours / resample).
#			- Skips rows containing NaNs/Infs so the CSV stays clean for ML loaders
#	"""
#	def __init__(self, path: Path, N: int):
#		self.path = path
#		self.N = N
#		self._f = open(self.path, "w", encoding="utf-8")
#		self._f.write(",".join(["t0"]+[f"x{i}" for i in range(1,N+1)]+["label"])+"\n")
#    
#	def write_row(self, t0_iso, x_vals, label):
#		if len(x_vals)!=self.N: 
#			return
#		if np.isnan(x_vals).any(): 
#			return
#		parts = [t0_iso]+[f"{v:.10g}" for v in x_vals]+[label]
#		self._f.write(",".join(parts)+"\n")
#
#	def close(self): 
#		self._f.close()
	
	
class FlatCSVWriter:
	"""
		Class to write flare time series data to CSV 
			- ratio-only:  t0, r1..rN, label
			- with hist:   t0, r1..rN, h1..hN, label
	"""
	def __init__(self, path: Path, N: int, include_history: bool = False):
		self.path = path
		self.N = N
		self.include_history = include_history
		self.expected = N * (2 if include_history else 1)
		self._f = open(self.path, "w", encoding="utf-8")

		ratio_cols = [f"r{i}" for i in range(1, N+1)]
		if self.include_history:
			hist_cols = [f"h{i}" for i in range(1, N+1)]
			cols = ["t0"] + ratio_cols + hist_cols + ["label"]
		else:
			cols = ["t0"] + ratio_cols + ["label"]
		self._f.write(",".join(cols) + "\n")

	def write_row(self, t0_iso, x_vals, label):
		# x_vals is either ratio (N) or ratio+hist (2N), depending on include_history
		if len(x_vals) != self.expected:
			return
		if np.isnan(x_vals).any():
			return
		parts = [t0_iso] + [f"{float(v):.10g}" for v in x_vals] + [label]
		self._f.write(",".join(parts) + "\n")

	def close(self):
		self._f.close()
        	
		
		
def build_history_channel(index, args, events_df, encoding="binary"):
	"""
		Returns a pd.Series H(t) indexed by `index` (1-min timestamps).
		H(t)=0 outside flares; during [t_start, t_end] either 1 (binary) or class code (ordinal).
		If multiple flares overlap, take the max at each timestamp.
	"""
    
	H = pd.Series(0.0, index=index, dtype="float32")
	if events_df.empty:
		return H

	# Class → ordinal mapping (A,B,C,M,X). Adjust if you don’t use A/B.
	if args.skip_ABflares_in_history:
		class_map = {"A": 0, "B": 0, "C": 1, "M": 2, "X": 3}
	else:
		class_map = {"A": 1, "B": 2, "C": 3, "M": 4, "X": 5}

	# We only need events overlapping the current input window
	# (this is the segment you already computed as `seg`).
	# Use the same `start`/`end` that define the input window.
	# Expect columns: timestamp_start, timestamp_end, flare_type
	for _, row in events_df.iterrows():
		s = row.get("timestamp_start")
		e = row.get("timestamp_end")
		if pd.isna(s) or pd.isna(e):
			continue

		# fast window overlap check
		if e < index[0] or s > index[-1]:
			continue

		# skip low type flares?
		ftype = str(row.get("flare_type", "")).strip()
		key = ftype[:1].upper() # leading letter (A/B/C/M/X)
		low_flare_type= (key=="A" or key=="B")
		if args.skip_ABflares_in_history and low_flare_type:
			continue

		val = 1.0
		if encoding == "ordinal":
			#ftype = str(row.get("flare_type", "")).strip()
			# leading letter (A/B/C/M/X)
			#key = ftype[:1].upper()
			val = float(class_map.get(key, 1))

		# mark overlap region on the same 1-min grid
		mask = (index >= s) & (index <= e)
		if mask.any():
			# take max in case multiple flares overlap
			H.loc[mask] = np.maximum(H.loc[mask].values, val)

	return H


def compute_series_for_sat(sat, args, events_df, flat_writer):
	""" 
		Create time series data
			1) Open & resample science flux
					- Load NetCDF → DataFrame → set index to time → resample to --resample (mean; default 1 min).
			2) Read daily backgrounds
					- Returns one timestamp per day with its background level
			3) For each background day:
					- Define the window [start, start+window).
					- Slice the science series and compute coverage; skip if < min_coverage.
					- Interpolate small gaps, then compute ratio = science/background.
					- Labeling: search flares in the next window (end, end+window], and pick X/M/C/NONE.
					- Save an .npz with two arrays:
							time: int64 nanoseconds since epoch (index),
							ratio: float ratio values.
					- If --emit-flat-csv and the window has exactly N points, write one CSV row:
							t0 = start (UTC ISO)
							x1 ... xN = ratio values
							label
	"""
    
	# - Set dir names
	sat_name = f"goes{sat}"
	sci_dir = args.science_root / sat_name
	bg_dir  = args.background_root / sat_name
	if not sci_dir.exists() or not bg_dir.exists():
		return pd.DataFrame()
		
	# - Set skip date range
	skip_ranges = []
	for rng in args.skip_date_ranges:
		try:
			start_str, end_str = rng.split(":")
			start_dt = pd.to_datetime(start_str + "-01", utc=True)
			# Use month end for end_dt
			year, month = map(int, end_str.split("-"))
			end_dt = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(1)
			skip_ranges.append((start_dt, end_dt))
		except Exception as e:
			raise ValueError(f"Invalid skip-date-range format '{rng}'. Expected YYYY-MM:YYYY-MM") from e

	# - Open science flux data
	print(f"INFO: Reading science flux data in dir {sci_dir} ...")
	ds = open_science_files(sci_dir, args.science_file_glob)
	tname = args.time_var or discover_time_name(ds)
	vname = args.science_var or discover_science_var(ds, args.band)
	sci = ds[[vname]].to_dataframe().reset_index().rename(columns={tname:"timestamp", vname:"science"})
	sci["timestamp"] = pd.to_datetime(sci["timestamp"], utc=True)
	sci = sci.set_index("timestamp").resample(args.resample).mean()
    
	# - Open background flux data    
	print(f"INFO: Reading background flux data in dir {bg_dir} ...")
	bg = read_background_dir(bg_dir, args.background_file_glob, args.band, args.background_var)
    
	# - Create time series
	window = pd.Timedelta(hours=args.window_hours)
	one = pd.Timedelta(args.resample)
	N_expected = int(window/one)
	print(f"Creating time series of length {N_expected} from science/bkg data ...")    

	N_skip_date= 0
	N_skip_year= 0
	N_skip_nan= 0
	N_skip_ftype= 0
	N_skip_badbkg= 0
	N_tot= 0

	rows = []
	outdict_list= []
	
	for _, row in bg.iterrows():
		start = pd.to_datetime(row["timestamp"], utc=True)
		end = start + window
		seg = sci.loc[start:end-one].copy()
		print(f"--> t_bkg=[{start}, {end}], seg: {seg}")
		N_tot+= 1

		# - Skip year?
		if start.year in set(args.skip_years):
			print(f"Skipping series: start {start} falls in skip years ({args.skip_years}) ...")
			N_skip_year+= 1
			continue
		
		# - Skip windows that start in any excluded date range
		if skip_ranges and any(s_dt <= start <= e_dt for (s_dt, e_dt) in skip_ranges):
			print(f"Skipping series: start {start} falls in a skip range ...")
			N_skip_date+= 1
			continue

		# - Check if time series has too any NANs
		frac_valid = seg["science"].notna().mean()    
		if seg.empty or frac_valid < args.min_coverage:
			print(f"WARN: Skipping time series as empty or with too many NANs (frac_valid={frac_valid}) ...")
			N_skip_nan+= 1
			continue
			
		# - Fill any NaNs by interpolation
		seg["science"] = seg["science"].interpolate(limit_direction="both")
		
		# - Skip time series if background measurement is 0 or inf/NaN
		if not np.isfinite(row["background"]) or row["background"]<=0: 
			print("WARN: Skipping time series as bkg value is 0/inf/nan...")
			N_skip_badbkg+= 1
			continue
			
		# - Skip time series that have flares occurring in the same time period?	
		if args.skip_inwindow_atleast != "NONE":
			thr = SEVERITY[args.skip_inwindow_atleast]

			# flares whose occurrence time falls inside the INPUT window
			inwin = events_df[(events_df["t_occ"] >= start) & (events_df["t_occ"] <= end)]
			if not inwin.empty:
				# use precomputed 'sev' if present; else map now
				sev_series = inwin["sev"] if "sev" in inwin.columns else inwin["class"].map(SEVERITY)
				if sev_series.max() >= thr:
					# skip this series entirely
					print(f"Skipping this series as it contains flares with type ({sev_series.max()}) above the threshold {thr} ...")
					N_skip_ftype+= 1
					continue

		# - Compute flux ratios
		ratio = seg["science"]/row["background"]
		
		# - Select flare events occurring in the range [end, end+window]
		#occ = events_df[(events_df["t_occ"]>end) & (events_df["t_occ"]<=end+window)]		
		gap = pd.Timedelta(hours=args.forecast_gap_hours)
		label_start = end + gap
		label_end   = end + gap + window
		occ = events_df[(events_df["t_occ"] > label_start) & (events_df["t_occ"] <= label_end)]

		label = choose_label(occ["class"].tolist())
		label_final= remap_label(label)
		id_final= LABEL2ID[label_final]
		
		# - Compute flare binary status time series?
		hist = None
		if args.emit_history_channel:
			print("Creating flare history time series channel ...")
			# Build only from events that overlap the input window
			# If you already have an `events_for_sat` filtered by satellite, reuse it here.
			hist = build_history_channel(seg.index, args, events_df, args.history_encoding)
    
		# - Save to npy
		#   NB: Converting each timestamp to nanoseconds since epoch (int64)
		#if not args.dry_run:
		#	out_path = args.outdir/f"series/{sat_name}/{start:%Y%m%d_%H%M%S}__W{args.window_hours}.npz"
		#	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
			
		#	np.savez_compressed(
		#		out_path,
		#		#time=seg.index.view("int64").values,
		#		time=seg.index.astype("int64").to_numpy(),
		#		ratio=ratio.values,
		#		label=np.array(label_final)  # label is a string like "NONE","C","M","X"
		#	)
			
		# ---- Save NPZ (ratio [+ hist]) ----
		if not args.dry_run:
			out_path = args.outdir / f"series/{sat_name}/{start:%Y%m%d_%H%M%S}__W{args.window_hours}.npz"
			Path(out_path).parent.mkdir(parents=True, exist_ok=True)
			print(f"Saving npy data to file {out_path} ...")
			
			# Base arrays
			npz_kwargs = {
				"time": seg.index.astype("int64").to_numpy(),        # ns since epoch
				"ratio": ratio.values.astype("float32"),
				"label": np.array(label_final)                        # e.g. "NONE","C","M","X"
			}

			# Optional history channel
			if args.emit_history_channel and (hist is not None):
				npz_kwargs["hist"] = hist.values.astype("float32")

			np.savez_compressed(out_path, **npz_kwargs)
			
		# - Save to CSV
		#if args.emit_flat_csv and flat_writer and len(ratio)==N_expected:
		#	flat_writer.write_row(start.strftime("%Y-%m-%d %H:%M:%S"), ratio.values, label_final)
			
		# ---- Save flat CSV (r1..rN [, h1..hN], label) ----
		if args.emit_flat_csv and flat_writer and len(ratio) == N_expected:
			start_str = start.strftime("%Y-%m-%d %H:%M:%S")

			if args.emit_history_channel and (hist is not None):
				# Concatenate ratio and history into a single 1D array
				flat_vals = np.concatenate([ratio.values, hist.values]).astype("float32")
			else:
				flat_vals = ratio.values.astype("float32")

			print("Saving csv data ...")
			flat_writer.write_row(start_str, flat_vals, label_final)
			
		# ---- Append to json output ----
		t_data_start= start.strftime("%Y-%m-%d %H:%M:%S")
		t_data_end= end.strftime("%Y-%m-%d %H:%M:%S")
		t_data_start_posix= start.timestamp() # number of seconds since Unix epoch (January 1, 1970)
		t_data_end_posix= end.timestamp() # number of seconds since Unix epoch (January 1, 1970)
		dt= one.total_seconds()
		t_forecast_start= label_start.strftime("%Y-%m-%d %H:%M:%S")
		t_forecast_end= label_end.strftime("%Y-%m-%d %H:%M:%S")
		t_forecast_start_posix= label_start.timestamp() # number of seconds since Unix epoch (January 1, 1970)
		t_forecast_end_posix= label_end.timestamp() # number of seconds since Unix epoch (January 1, 1970)
		xrf_flux_ratio_data= ratio.values.astype("float32")
		
		outdict= {
			"satellite": sat_name,
			"id": int(id_final),
			"label": str(label_final),
			"flare_type": str(label_final),
      "flare_id": int(id_final),
      "n_points": int(len(ratio)),
      "t_start": float(t_data_start_posix),
      "t_end": str(t_data_end_posix),
      "date_start": str(t_data_start),
      "date_end": str(t_data_end),
      "dt": float(dt),
      "date_forecast_start": str(t_forecast_start),
      "date_forecast_end": str(t_forecast_end),
      "t_forecast_start": float(t_forecast_start_posix),
      "t_forecast_end": float(t_forecast_end_posix),
      "xrs_flux_ratio": list(xrf_flux_ratio_data) 	
		}
		
		if args.emit_history_channel and (hist is not None):
			flare_hist_data= hist.values.astype("float32")
			outdict["flare_hist"]= list(flare_hist_data)
		
		print("Saving json data ...")
		outdict_list.append(outdict)
		
		# ---- Index/metadata row ----     
		rows.append(
			{
				"satellite": sat_name,
				"start_time": start,
				"end_time": end,
				"label": label_final,
				"n_points": int(len(ratio)),
				"has_hist": bool(args.emit_history_channel and (hist is not None))
			}
		)


	print("== STATS ==")
	print(f"Ntot= {N_tot}")
	print(f"Nskip_date= {N_skip_date}")
	print(f"Nskip_year= {N_skip_year}")
	print(f"Nskip_badbkg= {N_skip_badbkg}")
	print(f"Nskip_nan= {N_skip_nan}")
	print(f"Nskip_ftype= {N_skip_ftype}")

	return pd.DataFrame(rows), outdict_list


##################################
##      MAIN
##################################
def main():
	""" Main block """
    
	# - Parse arguments
	args = parse_args()
	args.outdir.mkdir(parents=True, exist_ok=True)
	if args.flat_csv_path is None:
		args.flat_csv_path = args.outdir/"series_flat.csv"
	
	# - Load event data
	events_df = load_events(args.events_file)
	
	# - Initialize flat writer
	flat_writer = None
	N= int(pd.Timedelta(hours=args.window_hours)/pd.Timedelta(args.resample))
	print(f"INFO: Initializing flat writer (h={args.window_hours}, N={N}) ...") 
	if args.emit_flat_csv:
		flat_writer= FlatCSVWriter(
			args.flat_csv_path, 
			N,
			include_history=args.emit_history_channel
		)  
	
	# - Prepare time series data
	print("INFO: Preparing time series data ...") 
	all_rows = []
	outdata_json= {"data": []}
	for sat in args.satellites:
		df, outdict_list = compute_series_for_sat(sat, args, events_df, flat_writer)
		if not df.empty: 
			all_rows.append(df)
		if outdict_list:
			outdata_json["data"].extend(outdict_list)
	
	# - Clear stuff
	if flat_writer: 
		print("INFO: Close flat writer ...") 
		flat_writer.close()
		
	if all_rows:
		pd.concat(all_rows).to_csv(args.outdir/"index.csv", index=False)

	# - Save metadata output
	print(f"INFO: Saving json metadata to file {args.outfile_json} ...")
	out_path: Path = args.outfile_json
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(outdata_json, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
	main()
