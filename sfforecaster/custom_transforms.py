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
from typing import List, Tuple, Sequence, Union, Optional

# - TORCH
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

PILImageTypes = tuple()  # filled lazily to avoid hard PIL dependency at import
try:
	from PIL import Image
	PILImageTypes = (Image.Image,)
except Exception:
	pass

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
##     RandomCenterCrop transform
##########################################
def _center_crop_params(h: int, w: int, frac: float) -> Tuple[int, int, int, int]:
	"""Return (top, left, new_h, new_w) for a center crop with the given scale fraction."""
	new_h = max(1, int(round(h * frac)))
	new_w = max(1, int(round(w * frac)))
	top = max(0, (h - new_h) // 2)
	left = max(0, (w - new_w) // 2)
	return top, left, new_h, new_w

def _is_tensor_image(x: torch.Tensor) -> bool:
	# Accept (C,H,W) or (H,W) tensors
	return torch.is_tensor(x) and x.ndim in (2, 3)
    
def _resize_image(img, size: int):
	if isinstance(img, PILImageTypes):
		return TF.resize(img, size=(size, size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
	if _is_tensor_image(img):
		return TF.resize(img, size=(size, size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
	raise TypeError(f"Unsupported image type: {type(img)}")
    		
def _crop_image(img, top: int, left: int, height: int, width: int):
	# Works for PIL or torch Tensor
	if isinstance(img, PILImageTypes):
		return TF.crop(img, top, left, height, width)
	if _is_tensor_image(img):
		# torchscript-compatible crop
		return img[..., top:top+height, left:left+width]
	raise TypeError(f"Unsupported image type: {type(img)}")    		
    		
class RandomCenterCrop(torch.nn.Module):
	"""
		Center crop with a random scale fraction in [min_frac, max_frac], keeping aspect ratio.
		Optionally resizes back to a square 'output_size'.
		Works on a single image (PIL or Tensor (C,H,W)).
	"""
	def __init__(
		self,
		min_frac: float = 0.7,
		max_frac: float = 1.0,
		output_size: Optional[int] = None,
		generator: Optional[torch.Generator] = None,
	):
		super().__init__()
		assert 0.0 < min_frac <= max_frac <= 1.0
		self.min_frac = float(min_frac)
		self.max_frac = float(max_frac)
		self.output_size = output_size
		self.generator = generator  # for reproducibility if you pass a seeded generator

	def _rand_uniform(self) -> float:
		if self.generator is None:
			return random.random()
		# Torch generator path (stable across Python processes if seeded)
		return float(torch.rand((), generator=self.generator).item())

	def forward(self, img: Union[torch.Tensor, "Image.Image"]):
		if isinstance(img, PILImageTypes):
			w, h = img.size
		elif _is_tensor_image(img):
			# (C,H,W)
			h, w = img.shape[-2], img.shape[-1]
		else:
			raise TypeError(f"Unsupported image type: {type(img)}")

		frac = self.min_frac + (self.max_frac - self.min_frac) * self._rand_uniform()
		top, left, new_h, new_w = _center_crop_params(h, w, frac)

		img = _crop_image(img, top, left, new_h, new_w)
		if self.output_size is not None:
			img = _resize_image(img, self.output_size)

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


class VideoRandomCenterCrop(torch.nn.Module):
	"""
		Apply the SAME random center crop to all frames in a video.
		Accepts:
			- Tensor of shape (T, C, H, W)  or (C, T, H, W)  -> set channels_first_time_dim accordingly
			- Sequence[List/PIL/Tensor] of frames (PIL or Tensor(C,H,W))
		Optionally resizes frames back to square 'output_size'.
	"""
	def __init__(
		self,
		min_frac: float = 0.7,
		max_frac: float = 1.0,
		output_size: Optional[int] = None,
		channels_first_time_dim: bool = True,  # True for (T,C,H,W). If (C,T,H,W), set False.
		generator: Optional[torch.Generator] = None,
	):
		super().__init__()
		assert 0.0 < min_frac <= max_frac <= 1.0
		self.min_frac = float(min_frac)
		self.max_frac = float(max_frac)
		self.output_size = output_size
		self.channels_first_time_dim = channels_first_time_dim
		self.generator = generator

	def _rand_uniform(self) -> float:
		if self.generator is None:
			return random.random()
		return float(torch.rand((), generator=self.generator).item())

	def forward(self, clip: Union[torch.Tensor, Sequence[Union[torch.Tensor, "Image.Image"]]]):
		# Determine (H,W) from first frame
		if torch.is_tensor(clip):
			if self.channels_first_time_dim:
				# (T, C, H, W)
				if clip.ndim != 4:
					raise ValueError("Expected video tensor with shape (T,C,H,W)")
				T, C, H, W = clip.shape
				h, w = H, W
			else:
				# (C, T, H, W)
				if clip.ndim != 4:
					raise ValueError("Expected video tensor with shape (C,T,H,W)")
				C, T, H, W = clip.shape
				h, w = H, W
		else:
			# Sequence of frames (PIL or Tensor)
			if len(clip) == 0:
				return clip
			f0 = clip[0]
			if isinstance(f0, PILImageTypes):
				w, h = f0.size
			elif _is_tensor_image(f0):
				h, w = f0.shape[-2], f0.shape[-1]
			else:
				raise TypeError(f"Unsupported frame type: {type(f0)}")

		# Sample ONE fraction, compute crop once
		frac = self.min_frac + (self.max_frac - self.min_frac) * self._rand_uniform()
		top, left, new_h, new_w = _center_crop_params(h, w, frac)

		# Apply to all frames
		if torch.is_tensor(clip):
			if self.channels_first_time_dim:
				frames = []
				for t in range(clip.shape[0]):
					ft = _crop_image(clip[t], top, left, new_h, new_w)
					if self.output_size is not None:
						ft = _resize_image(ft, self.output_size)
					frames.append(ft)
				out = torch.stack(frames, dim=0)  # (T,C,H,W)
			else:
				frames = []
				for t in range(clip.shape[1]):
					ft = _crop_image(clip[:, t], top, left, new_h, new_w)  # (C,H,W) view
					if self.output_size is not None:
						ft = _resize_image(ft, self.output_size)
					frames.append(ft)
				out = torch.stack(frames, dim=1)  # (C,T,H,W)
			return out
		else:
			out_frames = []
			for f in clip:
				f2 = _crop_image(f, top, left, new_h, new_w)
				if self.output_size is not None:
					f2 = _resize_image(f2, self.output_size)
				out_frames.append(f2)
			return out_frames
