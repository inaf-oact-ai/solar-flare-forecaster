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

# - TORCH
import torch


##############################
###   IMAGE LOAD
##############################
def load_img_for_inference(
	dataset, idx, 
	processor=None, 
	do_resize=False, 
	do_normalize=False,
	do_rescale=False
):
	""" Load image data for inference """
	
	# - Load image from dataset
	#   NB: This returns a Tensor of Shape [C,H,W] with transforms applied (if dataset has transform)
	input_tensor= dataset.load_image(idx)
	
	# - Apply model image processor?
	if processor is not None:	
		proc_out = processor(
			input_tensor,
			return_tensors="pt",
			do_resize=do_resize,       # set False if already resized
			do_normalize=do_normalize, # set False if already normalized
			do_rescale=do_rescale,     # set False if already scaled
		)
		pixel_values = proc_out["pixel_values"]  # This has already the batch dim
	
	else:
		# - Add batch dim for inference
		pixel_values= input_tensor.unsqueeze(0)
	
	return pixel_values
	
##############################
###   VIDEO LOAD
##############################
def load_video_for_inference(
	dataset, idx, 
	processor=None, 
	do_resize=False, 
	do_normalize=False,
	do_rescale=False
):
	""" Load video for inference """
	
	# - Load video frames
	#   NB: This returns a list of T tensor of Shape [C,H,W] with transforms applied (if dataset has transform)
	input_frames= dataset.load_video(idx)
	
	# - Add batch dim as 2D list
	videos= [input_frames]

	# - Set pixel values ---
	if processor is not None:
		# - NB: This works only with list of list of [C,H,W]
		proc_out = processor(
			videos,                      # list of length B; each item is list of T HWC frames
			return_tensors="pt",
			do_resize=do_resize,        # set False if you already resized
			do_normalize=do_normalize,  # set False if you already normalized
			do_rescale=do_rescale,      # set False if already rescaled
		)
		pixel_values = proc_out["pixel_values"].float()  # Shape: [B,T,C,H,W]
			
	else:
		# - Convert 2D list to tensor of Shape: [B,T,C,H,W]
		vids_tchw = [torch.stack(item, dim=0) for item in videos]
		pixel_values = torch.stack(vids_tchw, dim=0).float()
		
	return pixel_values
	
###################################
##  ORDINAL HELPERS
###################################
def coral_logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
	"""
		Convert CORAL/cumulative logits (K-1,) or (1, K-1) to K-class probs (K,).
		Works for any K >= 2. For K=4: logits = [>=C, >=M, >=X].
	"""
	if logits.dim() == 2:
		logits = logits.squeeze(0)   # (K-1,)
	Km1 = logits.shape[0]
	K = Km1 + 1

	p_ge = torch.sigmoid(logits)     # (K-1,)

	# optional: enforce non-increasing p_ge to avoid tiny numerical violations
	for i in range(1, Km1):
		p_ge[i] = torch.minimum(p_ge[i], p_ge[i-1])

	# K-class probabilities from cumulative probs
	comps = [1.0 - p_ge[0]]                         # P(class 0)
	for k in range(1, Km1):
		comps.append(p_ge[k-1] - p_ge[k])           # P(class k)
	comps.append(p_ge[Km1-1])                       # P(class K-1)

	probs = torch.stack(comps)                      # (K,)
	probs = torch.clamp(probs, min=1e-12)
	probs = probs / probs.sum()
	return probs  # (K,)

def coral_decode_with_thresholds(logits: torch.Tensor, thresholds=None) -> int:
	"""
		Count thresholds passed: class = sum( sigmoid(logit_k) >= t_k ).
		Defaults to t_k = 0.5 if thresholds is None.
	"""
	if logits.dim() == 2:
		logits = logits.squeeze(0)
	p_ge = torch.sigmoid(logits)
	if thresholds is None:
		ge = (p_ge >= 0.5).int()
	else:
		thr = torch.as_tensor(thresholds, dtype=logits.dtype, device=logits.device)
		ge = (p_ge >= thr).int()
	return int(ge.sum().item())


