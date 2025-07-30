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
