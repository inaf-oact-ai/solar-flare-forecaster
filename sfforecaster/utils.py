#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging
import io
import re

## COMMAND-LINE ARG MODULES
import getopt
import argparse
import collections
import csv
import json
import pickle

## ASTRO/IMG PROCESSING MODULES
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
import skimage
from PIL import Image

## TORCH MODULES
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

## DRAW MODULES
import matplotlib.pyplot as plt

## LOGGER
from sfforecaster import logger


##########################
##   MODEL UTILS
##########################
def extract_layer_id(name: str) -> int:
	""" Extract layer id from vision encoder layer name """ 
	match = re.search(r'\.layers\.(\d+)\.', name)
	if not match:
		logger.warning(f"No '.layers.<id>.' pattern found in: {name}")
		return -1
	return int(match.group(1))

##########################
##    DATA UTILS
##########################
def read_datalist(filename, key="data"):
	""" Read data json file """
	f= open(filename, "r")
	datalist= json.load(f)[key]
	return datalist
	
##########################
##    READ IMAGE
##########################
def read_img(filename, nchans=1, norm_range=(0.,1.), resize=False, resize_size=224, apply_zscale=True, contrast=0.25, to_uint8=False, set_nans_to_min=False, verbose=False):
  """ Read fits image and returns a numpy array """

  # - Check if filename is str, otherwise try to load it as Bytes.io with pillow
  if isinstance(filename, str):
    # - Check filename
    if filename=="":
      return None

    file_ext= os.path.splitext(filename)[1]

    # - Read fits image?
    try:
      if file_ext=='.fits':
        data= fits.open(filename)[0].data
      else:
        image= Image.open(filename)
        data= np.asarray(image)
    except Exception as e:
      logger.error("Failed to read image %s (err=%s), returning None..." % (filename, str(e)))
      return None

  else:
    try:
      image= Image.open(filename)
      data= np.asarray(image)
    except Exception as e:
      logger.error("Failed to read input image as Bytes.io with PIL (err=%s)!" % (str(e)))
      return None

  if data is None:
    return None

  # - Apply transform
  data_transf= transform_img(
    data,
    nchans=nchans,
    norm_range=norm_range,
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=to_uint8,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  
  if data_transf is None:
    return None

  return data_transf.astype(float)
  #return data_transf	
	
##########################
##    IMG TRANSFORMS
##########################		
def resize_img(
  image,
  min_dim=None, max_dim=None, min_scale=None,
  mode="square",
  order=1,
  preserve_range=True,
  anti_aliasing=False
):
  """ Resize numpy array to desired size """

  # Keep track of image dtype and return results in the same dtype
  image_dtype = image.dtype
  image_ndims= image.ndim

  # - Default window (y1, x1, y2, x2) and default scale == 1.
  h, w = image.shape[:2]
  window = (0, 0, h, w)
  scale = 1
  if image_ndims==3:
    padding = [(0, 0), (0, 0), (0, 0)] # with multi-channel images
  elif image_ndims==2:
    padding = [(0, 0)] # with 2D images
  else:
    logger.error("Unsupported image ndims (%d), returning None!" % (image_ndims))
    return None

  crop = None

  if mode == "none":
    return image, window, scale, padding, crop

  # - Scale?
  if min_dim:
    # Scale up but not down
    scale = max(1, min_dim / min(h, w))

  if min_scale and scale < min_scale:
    scale = min_scale

  # Does it exceed max dim?
  if max_dim and mode == "square":
    image_max = max(h, w)
    if round(image_max * scale) > max_dim:
      scale = max_dim / image_max

  # Resize image using bilinear interpolation
  if scale != 1:
    image= skimage.transform.resize(
      image,
      (round(h * scale), round(w * scale)),
      order=order,
      mode="constant",
      cval=0, clip=True,
      preserve_range=preserve_range,
      anti_aliasing=anti_aliasing, anti_aliasing_sigma=None
    )

  # Need padding or cropping?
  if mode == "square":
    # Get new height and width
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad

    if image_ndims==3:
      padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)] # multi-channel
    elif image_ndims==2:
      padding = [(top_pad, bottom_pad), (left_pad, right_pad)] # 2D images
    else:
      logger.error("Unsupported image ndims (%d), returning None!" % (image_ndims))
      return None

    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

  elif mode == "pad64":
    h, w = image.shape[:2]
    # - Both sides must be divisible by 64
    if min_dim % 64 != 0:
      logger.error("Minimum dimension must be a multiple of 64, returning None!")
      return None

    # Height
    if h % 64 > 0:
      max_h = h - (h % 64) + 64
      top_pad = (max_h - h) // 2
      bottom_pad = max_h - h - top_pad
    else:
      top_pad = bottom_pad = 0

    # - Width
    if w % 64 > 0:
      max_w = w - (w % 64) + 64
      left_pad = (max_w - w) // 2
      right_pad = max_w - w - left_pad
    else:
      left_pad = right_pad = 0

    if image_ndims==3:
      padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    elif image_ndims==2:
      padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    else:
      logger.error("Unsupported image ndims (%d), returning None!" % (image_ndims))
      return None

    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

  elif mode == "crop":
    # - Pick a random crop
    h, w = image.shape[:2]
    y = random.randint(0, (h - min_dim))
    x = random.randint(0, (w - min_dim))
    crop = (y, x, min_dim, min_dim)
    image = image[y:y + min_dim, x:x + min_dim]
    window = (0, 0, min_dim, min_dim)

  else:
    logger.error("Mode %s not supported!" % (mode))
    return None

  return image.astype(image_dtype)
  
	
def get_zscaled_data(data, contrast=0.25):
	""" Apply sigma clipping to input data and return transformed data """

	# - Find NaNs pixels
	cond= np.logical_and(data!=0, np.isfinite(data))

	# - Apply zscale transform
	transform= ZScaleInterval(contrast=contrast)
	data_transf= transform(data)	

	# - Set NaNs to 0
	data_transf[~cond]= 0

	return data_transf
	
def transform_img(data, nchans=1, norm_range=(0.,1.), resize=False, resize_size=224, apply_zscale=True, contrast=0.25, to_uint8=False, set_nans_to_min=False, verbose=False):
  """ Transform input image data and return transformed data """

  # - Make copy
  data_transf= data.copy()

  # - Replace NANs pixels with 0 or min
  cond_nonan= np.isfinite(data_transf)
  cond_nonan_noblank= np.logical_and(data_transf!=0, np.isfinite(data_transf))
  data_1d= data_transf[cond_nonan_noblank]
  if data_1d.size==0:
    logger.warn("Input data are all zeros/nan, return None!")
    return None

  if set_nans_to_min:
    data_transf[~cond_nonan]= data_min
  else:
    data_transf[~cond_nonan]= 0

  if verbose:
    print("== DATA MIN/MAX (BEFORE TRANSFORM)==")
    print(data_transf.min())
    print(data_transf.max())

	# - Apply zscale stretch?
  if apply_zscale:
    transform= ZScaleInterval(contrast=contrast)
    data_zscaled= transform(data_transf)
    data_transf= data_zscaled

  # - Resize image?
  if resize:
    interp_order= 3 # 1=bilinear, 2=biquadratic, 3=bicubic, 4=biquartic, 5=biquintic
    data_transf= resize_img(
      data_transf,
      min_dim=resize_size, max_dim=resize_size, min_scale=None,
      mode="square",
      order=interp_order,
      preserve_range=True,
      anti_aliasing=False
    )

  if verbose:
    print("== DATA MIN/MAX (AFTER TRANSFORM) ==")
    print(data_transf.shape)
    print(data_transf.min())
    print(data_transf.max())

  # - Apply min/max normalization
  data_min= data_transf.min()
  data_max= data_transf.max()
  norm_min= norm_range[0]
  norm_max= norm_range[1]
  data_norm= (data_transf-data_min)/(data_max-data_min) * (norm_max-norm_min) + norm_min
  data_transf= data_norm

  if verbose:
    print("== DATA MIN/MAX (AFTER TRANSFORM) ==")
    print(data_transf.shape)
    print(data_transf.min())
    print(data_transf.max())

  # - Expand 2D data to desired number of channels (if>1): shape=(ny,nx,nchans)
  ndim= data_transf.ndim
  if nchans>1 and ndim==2:
    data_transf= np.stack((data_transf,) * nchans, axis=-1)
    
  # - For 3D data, check number of channels, eventually copying last channel in new ones
  if ndim==3:  	
    nchans_curr= data_transf.shape[-1]

    if nchans_curr!=nchans:
      data_resized= np.zeros((data_transf.shape[0], data_transf.shape[1], nchans))

      expanding= (nchans>nchans_curr)
      if expanding:
        for i in range(nchans):
          if i<nchans_curr:
            data_resized[:,:,i]= data_transf[:,:,i]
          else:
            data_resized[:,:,i]= data_transf[:,:,nchans_curr-1]	
      else:
        for i in range(nchans):
          data_resized[:,:,i]= data_transf[:,:,i]
			
      data_transf= data_resized

  # - Convert to uint8
  if to_uint8:
    data_transf= data_transf.astype(np.uint8)

  if verbose:
    print("== DATA MIN/MAX (AFTER RESHAPE) ==")
    print(data_transf.shape)
    print(data_transf.min())
    print(data_transf.max())

  return data_transf
  
##########################
##    IMG PIL LOADERS
##########################	
def load_img_as_pil_float(filename, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Convert numpy array to PIL float image norm to [0,1] """

  # - Read FITS from file and get transformed npy array
  data= read_img(
    filename,
    nchans=1,
    norm_range=(0.,1.),
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=False,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  if data is None:
    logger.warn("Read image is None!")
    return None

  # - Convert to PIL image
  return Image.fromarray(data)	
  
def load_img_as_pil_rgb(filename, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Convert numpy array to PIL 3chan RGB image norm to [0,255], uint8 """

  # - Read FITS from file and get transformed npy array
  data= read_img(
    filename,
    nchans=3,
    norm_range=(0.,255.),
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=False,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  if data is None:
    logger.warn("Read image is None!")
    return None
    
  # - Convert to PIL RGB image (NB: with 3 chan fromarray requires uint18 0-255 data)
  return Image.fromarray(data.astype(np.uint8)).convert("RGB")
	
##########################
##    IMG NUMPY LOADERS
##########################	
def load_img_as_npy_float(filename, add_chan_axis=True, add_batch_axis=True, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Return numpy float image array norm to [0,1] """

  # - Read FITS from file and get transformed npy array
  data= read_img(
    filename,
    nchans=1,
    norm_range=(0.,1.),
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=False,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  if data is None:
    logger.warn("Read image is None!")
    return None

  # - Add channel axis if missing?
  ndim= data.ndim
  if ndim==2 and add_chan_axis:
    data_reshaped= np.stack((data,), axis=-1)
    data= data_reshaped

    # - Add batch axis if requested
    if add_batch_axis:
      data_reshaped= np.stack((data,), axis=0)
      data= data_reshaped

  return data.astype(float)
  
def load_img_as_npy_rgb_float(filename, add_chan_axis=True, add_batch_axis=True, resize=False, resize_size=224, apply_zscale=True, contrast=0.25, set_nans_to_min=False, verbose=False):
  """ Return numpy float image 3-chan array norm to [0,1] """

  # - Read FITS from file and get transformed npy array
  data= read_img(
    filename,
    nchans=3,
    norm_range=(0.,1.),
    resize=resize, resize_size=resize_size,
    apply_zscale=apply_zscale, contrast=contrast,
    to_uint8=False,
    set_nans_to_min=set_nans_to_min,
    verbose=verbose
  )
  if data is None:
    logger.warn("Read image is None!")
    return None

  # - Add batch axis if requested
  if add_batch_axis:
    data_reshaped= np.stack((data,), axis=0)
    data= data_reshaped

  return data

