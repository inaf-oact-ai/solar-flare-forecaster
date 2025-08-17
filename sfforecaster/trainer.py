#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# - TORCH
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F

# - TRANSFORMERS
import transformers
from transformers import Trainer
from transformers import pipeline
from transformers import AutoProcessor, AutoModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers import EvalPrediction    
import evaluate

# - SCLASSIFIER-VIT
from sfforecaster.utils import *
from sfforecaster import logger

##########################################
##    FOCAL LOSS
##########################################
class FocalLossMultiClass(nn.Module):
	"""
		Multi-class focal loss with logits.
			- alpha: Tensor [C] or float or None (class weighting)
			- gamma: focusing parameter
	"""

	def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
		super().__init__()
		self.gamma = gamma
		self.reduction = reduction
		self.alpha = alpha  # None | float | Tensor[C]

	def forward(self, logits, targets):
		# logits: [B, C], targets: [B] int64
		log_probs = F.log_softmax(logits, dim=1)              # [B, C]
		probs = torch.exp(log_probs)                          # [B, C]
		# pick the prob/log_prob of the target class
		pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # [B]
		log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
		focal_term = (1.0 - pt).clamp_min(1e-8).pow(self.gamma)      # [B]

		if self.alpha is None:
			alpha_t = 1.0
		elif isinstance(self.alpha, float):
			alpha_t = self.alpha
		else:
			# alpha is tensor [C]
			alpha_t = self.alpha.to(logits.device).gather(0, targets)

		loss = -alpha_t * focal_term * log_pt  # [B]
		if self.reduction == "mean":
			return loss.mean()
		elif self.reduction == "sum":
			return loss.sum()
		else:
			return loss


class FocalLossMultiLabel(nn.Module):
	"""
		Multi-label focal loss with logits (BCE variant).
		pos_weight acts like class-wise alpha for positives.
	"""
    
	def __init__(self, gamma=2.0, pos_weight=None, reduction="mean"):
		super().__init__()
		self.gamma = gamma
		self.pos_weight = pos_weight  # Tensor [C] or None
		self.reduction = reduction

	def forward(self, logits, targets):
		# logits: [B, C], targets: [B, C] in {0,1}
		# stable BCE terms
		bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
		# pt for each class
		p = torch.sigmoid(logits)
		pt = torch.where(targets == 1, p, 1 - p)
		focal_term = (1 - pt).clamp_min(1e-8).pow(self.gamma)

		loss = focal_term * bce
		if self.reduction == "mean":
			return loss.mean()
		elif self.reduction == "sum":
			return loss.sum()
		else:
			return loss


##########################################
##    WEIGHTED-LOSS CUSTOM TRAINER
##########################################
class AdvancedImbalanceTrainer(Trainer):
	""" Custom trainer implementing a weighted loss and dataset weighted resampling to tackle class imbalance """
    
	def __init__(
		self,
		*args,
		class_weights=None,          # torch.tensor [C] or None
		multilabel=False,
		loss_type="ce",              # "ce" or "focal"
		focal_gamma=2.0,
		focal_alpha=None,            # None | float | tensor[C] (multiclass)
		sample_weights=None,         # list[float] per-example for train set or None
		**kwargs
	):
		super().__init__(*args, **kwargs)
		self.class_weights = class_weights
		self.multilabel = multilabel
		self.loss_type = loss_type
		self.focal_gamma = focal_gamma
		self.focal_alpha = focal_alpha
		self.sample_weights = sample_weights

		# - Build the criterion
		if self.multilabel:
			if self.loss_type == "focal":
				pos_w = self.class_weights.to(self.model.device) if self.class_weights is not None else None
				self.loss_fct = FocalLossMultiLabel(gamma=self.focal_gamma, pos_weight=pos_w, reduction="mean")
			else:
				pos_w = self.class_weights.to(self.model.device) if self.class_weights is not None else None
				self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
		else:
			if self.loss_type == "focal":
				alpha = self.focal_alpha
				if isinstance(alpha, torch.Tensor):
					alpha = alpha.to(self.model.device)
				self.loss_fct = FocalLossMultiClass(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
			else:
				w = self.class_weights.to(self.model.device) if self.class_weights is not None else None
				self.loss_fct = torch.nn.CrossEntropyLoss(weight=w)

	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		""" Override trainer compute_loss function """
		
		pixel_values = inputs.get("pixel_values")
		labels = inputs.get("labels")
		outputs = model(pixel_values)
		#labels = inputs.pop("labels")
		#outputs = model(**inputs)
		logits = outputs.logits

		if self.multilabel:
			labels = labels.float()
			loss = self.loss_fct(logits, labels)
		else:
			loss = self.loss_fct(logits, labels)

		return (loss, outputs) if return_outputs else loss

	def get_train_dataloader(self):
		""" Get train dataloader with resampling """
		
		if self.sample_weights is None:
			logger.info("No sample weights given, returning standard train dataloader ...")
			return super().get_train_dataloader()

		# - Weighted sampler (per-example) — replacement=True is standard here
		logger.info("Creating weighted random sampler ...")
		sampler = WeightedRandomSampler(
			weights=self.sample_weights,
			num_samples=len(self.sample_weights),
			replacement=True,
			generator=torch.Generator().manual_seed(self.args.seed)
		)
		return DataLoader(
			self.train_dataset,
			batch_size=self.args.train_batch_size,
			sampler=sampler,
			collate_fn=self.data_collator,
			num_workers=self.args.dataloader_num_workers,
			pin_memory=self.args.dataloader_pin_memory,
			drop_last=self.args.dataloader_drop_last,
		)
		
