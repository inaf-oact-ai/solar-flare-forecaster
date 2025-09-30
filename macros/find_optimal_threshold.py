#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import random
import numpy as np
import argparse
from pathlib import Path
import numpy as np, csv, os
from typing import List, Tuple, Dict, Optional

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	parser.add_argument('-inputfiles','--inputfiles', dest='inputfiles', required=True, type=str, help='Input data metric files separated by commas') 
	parser.add_argument('-min_recall', '--min_recall', dest='min_recall', required=False, type=float, default=0.3, action='store', help='Min recall filter (default=0.3)')
	parser.add_argument('-min_precision', '--min_precision', dest='min_precision', required=False, type=float, default=0.1, action='store', help='Min precision filter (default=0.3)')
	parser.add_argument('-penalty_std', '--penalty_std', dest='penalty_std', required=False, type=float, default=0.05, action='store', help='Penalty std parameter (default=0.05)')
		
	args, _unknown = parser.parse_known_args()
	
	return args	
	
def _load_curves_csv(path: str) -> Dict[str, np.ndarray]:
    # expects header: threshold,precision,recall,f1,tss,hss,mcc,apss
    arr = np.genfromtxt(path, delimiter=",", names=True)
    # Ensure 1D arrays even if file has a single row
    return {k: np.atleast_1d(arr[k]) for k in arr.dtype.names}

def consensus_tau_from_curve_csvs(
    csv_paths: List[str],
    grid: Optional[np.ndarray] = None,
    min_recall: Optional[float] = None,
    min_precision: Optional[float] = None,
    penalty_std: float = 0.0,          # e.g., 0.05 to slightly prefer stable τ
    weights: Optional[List[float]] = None  # optional run weights (e.g., by val size)
) -> Tuple[float, Dict[str, float]]:
    """
    Aggregate precision/recall/TSS/etc. vs threshold across runs and pick one τ*.

    Returns:
      tau_star, summary dict with mean/stats at tau_star.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 1001)

    metrics = ["precision", "recall", "f1", "tss", "hss", "mcc", "apss"]
    stacks = {m: [] for m in metrics}

    runs = [_load_curves_csv(p) for p in csv_paths]
    for r in runs:
        thr = r["threshold"]
        for m in metrics:
            # interpolate each metric to the common grid
            stacks[m].append(np.interp(grid, thr, r[m]))

    X = {m: np.stack(stacks[m], axis=0) for m in metrics}  # shape (R, T)

    # weights (default: uniform)
    R = X["tss"].shape[0]
    if weights is None:
        w = np.ones(R) / R
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (w.sum() + 1e-12)

    def wmean(A):  # A: (R, T)
        return np.tensordot(w, A, axes=(0, 0))

    # --- MEAN over runs for *all* metrics present in X
    mean = {m: wmean(X[m]) for m in metrics}                       # (T,)
    
    # --- STD over runs for the *same* metrics; zeros if only one run
    std = {}
    for m in X:
        if X[m].shape[0] > 1:
            std[m] = X[m].std(axis=0, ddof=1)
        else:
            std[m] = np.zeros_like(mean[m])
    
    print("mean")
    print(mean)
    
    print("std")
    print(std)
     
    # Ensure 'tss' exists in both; if not, fall back gracefully
    if "tss" not in mean:
        raise KeyError("TSS not found in aggregated curves; check CSV headers/columns.")
    if "tss" not in std:
        std["tss"] = np.zeros_like(mean["tss"])

    # optional constraints
    valid = np.ones_like(grid, dtype=bool)
    if min_recall is not None:
        valid &= (mean["recall"] >= float(min_recall))
    if min_precision is not None:
        valid &= (mean["precision"] >= float(min_precision))
    if not np.any(valid):
        # if constraints impossible, relax them
        print("WARNING: No contraint satisfied, relaxing all of them...")
        valid[:] = True

    # stability-aware score: mean TSS minus λ * std(TSS)
    score = mean["tss"] - float(penalty_std) * std["tss"]
    idx = int(np.nanargmax(np.where(valid, score, -np.inf)))
    tau_star = float(grid[idx])

    summary = {
        "tau_star": tau_star,
        "mean_tss": float(mean["tss"][idx]),
        "std_tss": float(std["tss"][idx]),
        "mean_precision": float(mean["precision"][idx]),
        "std_precision": float(std["precision"][idx]),
        "mean_recall": float(mean["recall"][idx]),
        "std_recall": float(std["recall"][idx]),
        "mean_hss": float(mean["hss"][idx]),
        "std_hss": float(std["hss"][idx]),
        "mean_mcc": float(mean["mcc"][idx]),
        "std_mcc": float(std["mcc"][idx]),
        "mean_f1": float(mean["f1"][idx]),
        "std_f1": float(std["f1"][idx]),
    }
    return tau_star, summary
    
##########################
###  MAIN
##########################
def main():
	""" Main method """
	
	# - Read args
	args= get_args()
	filenames= [str(x.strip()) for x in args.inputfiles.split(',')]

	# - Pick τ* that maximizes mean TSS across runs, while keeping mean recall ≥ 0.3
	tau_star, info = consensus_tau_from_curve_csvs(
		filenames, 
		min_recall=args.min_recall, 
		min_precision=args.min_precision,
		penalty_std=args.penalty_std
	)
	
	print("Consensus τ* =", tau_star)
	print(info)
    
	return 0
		
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())    
