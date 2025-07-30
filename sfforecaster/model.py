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
import torch.nn as nn
from transformers import VideoMAEForVideoClassification
from transformers import ViTForImageClassification

###########################################
###     MULTI-HORIZON VIDEO MODEL
###########################################
class MultiHorizonVideoMAE(VideoMAEForVideoClassification):
	""" Class to perform multi-horizon forecasting with VideoMAE """
	def __init__(
		self, 
		config, 
		num_horizons=3, 
		num_classes=4
	):
		super().__init__(config)
		self.num_horizons = num_horizons
		self.num_classes = num_classes
		hidden_size = config.hidden_size

		# - Dynamically create one head per horizon
		self.classifiers = nn.ModuleList(
			[nn.Linear(hidden_size, num_classes) for _ in range(num_horizons)]
		)

	def forward(self, pixel_values, labels=None, output_hidden_states=False):
		outputs = self.videomae(pixel_values)
		pooled_output = outputs[0]  # CLS token

		# Compute logits for each horizon
		logits = [head(pooled_output) for head in self.classifiers]  # List of [B, num_classes]

		loss = None
		if labels is not None:
			# labels: shape [B, num_horizons] with class indices
			loss_fct = nn.CrossEntropyLoss()
			loss = sum(
				[loss_fct(logits[i], labels[:, i]) for i in range(self.num_horizons)]
			)

		return {
			"loss": loss,
			"logits": logits  # list of tensors [B, num_classes]
		}
		
###########################################
###     MULTI-HORIZON IMAGE MODEL
###########################################		
class MultiHorizonViT(ViTForImageClassification):
	""" Class to perform multi-horizon forecasting with ViT models """
	
	def __init__(self, config, num_horizons=3, num_classes=4):
		super().__init__(config)
		self.num_horizons = num_horizons
		self.num_classes = num_classes
		hidden_size = config.hidden_size

		# - Define N classification heads
		self.classifiers = nn.ModuleList(
			[nn.Linear(hidden_size, num_classes) for _ in range(num_horizons)]
		)

		# - Optional: remove the default classifier to avoid confusion
		del self.classifier

	def forward(self, pixel_values, labels=None):
		outputs = self.vit(pixel_values)
		pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

		logits = [head(pooled_output) for head in self.classifiers]  # List of [B, C]

		loss = None
		if labels is not None:
			# labels: [B, N] with class indices
			loss_fct = nn.CrossEntropyLoss()
			loss = sum(
				[loss_fct(logits[i], labels[:, i]) for i in range(self.num_horizons)]
			)

		return {
			"loss": loss,
			"logits": logits  # list of tensors [B, num_classes]
		}		
		
