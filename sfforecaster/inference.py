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
