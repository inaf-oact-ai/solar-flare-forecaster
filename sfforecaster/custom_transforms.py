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
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


##########################################
##    FlippingTransform
##########################################
class FlippingTransform(torch.nn.Module):
	"""Flipping: lr, ud or nothing"""

	def __init__(self):
		super().__init__()

	def forward(self, img):
		op= random.choice([1,2,3])
		if op==1:
			return TF.hflip(img)
		elif op==2:
			return TF.vflip(img)
		else:
			return img

##########################################
##     Rotate90 transform
##########################################
class Rotate90Transform(torch.nn.Module):
	"""Rotate by one of the given angles: 90, 270, """

	def __init__(self):
		super().__init__()

	def forward(self, img):
		op= random.choice([1,2,3,4])
		if op==1:
			return TF.rotate(img, 90)
		elif op==2:
			return TF.rotate(img, 180)
		elif op==3:
			return TF.rotate(img, 270)
		elif op==4:
			return img
			
##########################################
##    VIDEO TRANSFORMS
##########################################
def _as_tensor_TCHW(video):
	"""Accept (T,C,H,W) tensor or list[C,H,W] -> return (T,C,H,W) tensor and a flag to restore list."""
	if isinstance(video, (list, tuple)):
		restore_list = True
		video = torch.stack(video, dim=0)  # (T,C,H,W)
	else:
		restore_list = False
	if video.dim() != 4:
		raise ValueError(f"Expected video (T,C,H,W) or list of (C,H,W), got shape {tuple(video.shape)}")
	return video, restore_list

def _maybe_to_list(video_tensor, restore_list):
	return [f for f in video_tensor] if restore_list else video_tensor

def _ensure_float_0_1(x):
	# Make sure we can normalize; if byte, convert to float in [0,1].
	if x.dtype == torch.uint8:
		x = x.float() / 255.0
	elif not torch.is_floating_point(x):
		x = x.float()
	return x


class VideoFlipping(torch.nn.Module):
	"""Flipping: left-right, up-down, or nothing (chosen once per clip)."""

	def __init__(self):
		super().__init__()

	def forward(self, video):
		vid, return_list = _as_tensor_TCHW(video)  # (T, C, H, W)
		
		op = random.choice([1, 2, 3])  # 1=LR, 2=UD, 3=none
		if op == 1:
			# horizontal flip (flip width)
			vid = vid.flip(dims=[-1])
		elif op == 2:
			# vertical flip (flip height)
			vid = vid.flip(dims=[-2])
		# else: no-op

		return _maybe_to_list(vid, return_list)

class VideoResize(torch.nn.Module):
	def __init__(self, size, interpolation=InterpolationMode.BICUBIC, antialias=True):
		super().__init__()
		self.size = size
		self.interp = interpolation
		self.antialias = antialias

	def forward(self, video):
		vid, to_list = _as_tensor_TCHW(video)  # (T,C,H,W)
		T_, C, H, W = vid.shape
        
		# Apply with a single call by flattening time into batch
		vid = TF.resize(
			vid.reshape(-1, C, H, W),
			self.size,
			interpolation=self.interp,
			antialias=self.antialias,
		)
		vid = vid.reshape(T_, C, vid.shape[-2], vid.shape[-1])
		
		return _maybe_to_list(vid, to_list)
		
class VideoNormalize(torch.nn.Module):
	def __init__(self, mean, std, inplace=False):
		super().__init__()	
		# store as tensors for broadcasting
		self.register_buffer("mean", torch.tensor(mean)[None, :, None, None])
		self.register_buffer("std",  torch.tensor(std )[None, :, None, None])
		self.inplace = inplace

	def forward(self, video):
		vid, to_list = _as_tensor_TCHW(video)
		vid = _ensure_float_0_1(vid)
		if not self.inplace:
			vid = vid.clone()
		vid = (vid - self.mean.to(vid.device, vid.dtype)) / self.std.to(vid.device, vid.dtype)
		return _maybe_to_list(vid, to_list)


class VideoRotate90(torch.nn.Module):
	"""Rotate the whole clip by one of {90, 180, 270, none} degrees."""

	def __init__(self):
		super().__init__()

	def forward(self, video):
		vid, return_list = _as_tensor_TCHW(video)  # (T, C, H, W)

		op = random.choice([1, 2, 3, 4])  # 1=90, 2=180, 3=270, 4=none
		if op == 1:
			vid = torch.rot90(vid, 1, dims=(-2, -1))
		elif op == 2:
			vid = torch.rot90(vid, 2, dims=(-2, -1))
		elif op == 3:
			vid = torch.rot90(vid, 3, dims=(-2, -1))
		# op==4: no-op

		return _maybe_to_list(vid, return_list)

