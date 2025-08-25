#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import sys
import os
import random
import math
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
		#self.alpha = alpha  # None | float | Tensor[C]
		self.register_buffer("alpha", alpha if alpha is not None else None)  # stays on the right device 

	def forward(self, logits, targets):
		# - logits: [B, C], targets: [B] int64
		log_probs = F.log_softmax(logits, dim=1)              # [B, C]
		probs = torch.exp(log_probs)                          # [B, C]
		
		# - pick the prob/log_prob of the target class
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
##    SOLAR LOSSES
##########################################
def _acc(TN, FP, FN, TP):  # not used by default, but kept for completeness
	denom = TN + FP + FN + TP
	return torch.nan_to_num((TP + TN) / denom, nan=0.0)

def _prec(TN, FP, FN, TP):
	denom = FP + TP
	return torch.nan_to_num(TP / denom, nan=0.0)

def _rec(TN, FP, FN, TP):
	denom = FN + TP
	return torch.nan_to_num(TP / denom, nan=0.0)

def _spec(TN, FP, FN, TP):
	denom = FP + TN
	return torch.nan_to_num(TN / denom, nan=0.0)

def _f1(TN, FP, FN, TP):
	p = _prec(TN, FP, FN, TP)
	r = _rec(TN, FP, FN, TP)
	denom = p + r
	return torch.nan_to_num(2.0 * (p * r) / denom, nan=0.0)

def _tss(TN, FP, FN, TP):
	# TSS = TPR + TNR − 1 = recall + specificity − 1
	return _rec(TN, FP, FN, TP) + _spec(TN, FP, FN, TP) - 1.0

def _csi(TN, FP, FN, TP):
	denom = FN + FP + TP
	return torch.nan_to_num(TP / denom, nan=0.0)

def _hss1(TN, FP, FN, TP):
	denom = FN + TP
	return torch.nan_to_num((TP - FP) / denom, nan=0.0)

def _hss2(TN, FP, FN, TP):
	# 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(TN+FP))
	num = 2.0 * (TP * TN - FP * FN)
	denom = (TP + FN) * (FN + TN) + (TP + FP) * (TN + FP)
	return torch.nan_to_num(num / denom, nan=0.0)

_SCORE_MAP = {
	"accuracy": _acc,
	"precision": _prec,
	"recall": _rec,
	"specificity": _spec,
	"f1_score": _f1,
	"tss": _tss,
	"csi": _csi,
	"hss1": _hss1,
	"hss2": _hss2,
}

def _F_uniform(p: torch.Tensor) -> torch.Tensor:
	# uniform prior over threshold ⇒ F(p) = p
	return p

def _F_cosine(p: torch.Tensor, mu: float, delta: float) -> torch.Tensor:
	# Raised cosine CDF as in the TF code
	# piecewise: 0 for p < mu-delta; 1 for p > mu+delta; smoothed in between
	out = torch.zeros_like(p)
	left = mu - delta
	right = mu + delta

	# middle region
	mid_mask = (p >= left) & (p <= right)
	out[p > right] = 1.0

	# 0.5*(1 + (p-mu)/delta + 1/pi * sin(pi*(p-mu)/delta))
	pm = (p[mid_mask] - mu) / delta
	out[mid_mask] = 0.5 * (1.0 + pm + (1.0 / math.pi) * torch.sin(math.pi * pm))
	return out.clamp(0.0, 1.0)
	
	
class ScoreOrientedLoss(nn.Module):
	"""
		PyTorch implementation of SOL (Guastavino & Marchetti 2021).
		Defaults to uniform prior + score='tss' ⇒ loss ≈ -TSS (optionally + constant).

		Args:
			- score: one of ['accuracy','precision','recall','specificity','f1_score','tss','csi','hss1','hss2']
			- distribution: 'uniform' or 'cosine'
			- mu, delta: float or list[float] (used only for 'cosine'); length K if multiclass with class-wise params
			- mode: 'average' or 'weighted' (aggregation across one-vs-rest tasks in multiclass)
			- from_logits: if True, apply sigmoid (binary) or softmax (multiclass)
			- add_constant: if True, return `-score + 1`; if False, return `-score`
		Usage:
			# binary: y_true (B,), y_pred (B,) or (B,1)
			# multiclass: y_true (B,) int64 labels; y_pred (B,K)
	"""

	def __init__(
		self,
		score: str = "tss",
		distribution: str = "uniform",
		mu: Union[float, List[float]] = 0.5,
		delta: Union[float, List[float]] = 0.1,
		mode: str = "average",
		from_logits: bool = True,
		add_constant: bool = False,
	):
		super().__init__()
		assert score in _SCORE_MAP, f"Unknown score: {score}"
		assert distribution in ("uniform", "cosine")
		assert mode in ("average", "weighted")
		self.score_fn = _SCORE_MAP[score]
		self.distribution = distribution
		self.mu = mu
		self.delta = delta
		self.mode = mode
		self.from_logits = from_logits
		self.add_constant = add_constant

	def _apply_distribution(self, p: torch.Tensor, j: Optional[int] = None) -> torch.Tensor:
		if self.distribution == "uniform":
			return _F_uniform(p)
		else:
			# cosine; allow per-class mu/delta if list provided
			if isinstance(self.mu, list) or isinstance(self.mu, tuple):
				mu = float(self.mu[j])
			else:
				mu = float(self.mu)
			if isinstance(self.delta, list) or isinstance(self.delta, tuple):
				delta = float(self.delta[j])
			else:
				delta = float(self.delta)
			return _F_cosine(p, mu, delta)

	@torch.no_grad()
	def _confusion_expected(self, y_true_bin: torch.Tensor, Fp: torch.Tensor):
		"""
			Expected confusion for a binary task given:
			- y_true_bin: (B,) in {0,1}
			- Fp: (B,) in [0,1]  (prior-CDF evaluated at predicted prob)
			Returns TN, FP, FN, TP as scalars (tensors).
		"""
		y = y_true_bin.float()
		one_minus_y = 1.0 - y
		one_minus_Fp = 1.0 - Fp

		TN = torch.sum(one_minus_y * one_minus_Fp)
		TP = torch.sum(y * Fp)
		FP = torch.sum(one_minus_y * Fp)
		FN = torch.sum(y * one_minus_Fp)
		return TN, FP, FN, TP

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		"""
			y_pred:
				- binary: shape (B,) or (B,1)
				- multiclass: shape (B,K)
			y_true:
				- binary: (B,) in {0,1}
				- multiclass: (B,) int64 class indices
		"""
		if y_pred.dim() == 1 or y_pred.size(-1) == 1:
			# --- Binary case ---
			y_pred = y_pred.view(-1)
			if self.from_logits:
				p = torch.sigmoid(y_pred)
			else:
				p = y_pred.clamp(0.0, 1.0)
			y = y_true.view(-1).float()

			Fp = self._apply_distribution(p)  # uniform ⇒ F(p)=p
			TN, FP, FN, TP = self._confusion_expected(y, Fp)
			score = self.score_fn(TN, FP, FN, TP)
			loss = -score
			if self.add_constant: # TSS=[-1,1] --> LOSS=-TSS=[-1,1] --> LOSS=[0,2] 
				loss = loss + 1.0
			return loss

		else:
			# --- Multiclass one-vs-rest ---
			B, K = y_pred.shape
			if self.from_logits:
				p = F.softmax(y_pred, dim=-1)  # (B,K)
			else:
				p = y_pred.clamp(0.0, 1.0)
				p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)

			# y_true as one-hot (B,K), float
			if y_true.dtype != torch.long:
				y_true = y_true.long()
			y_oh = F.one_hot(y_true, num_classes=K).float()  # (B,K)

			scores = []
			weights = []  # for 'weighted' mode: #negatives per class
			for j in range(K):
				yj = y_oh[:, j]              # (B,)
				pj = p[:, j]                 # (B,)
				Fpj = self._apply_distribution(pj, j=j)
				TN, FP, FN, TP = self._confusion_expected(yj, Fpj)
				sj = self.score_fn(TN, FP, FN, TP)
				scores.append(sj)
				if self.mode == "weighted":
					# weight by #negatives to mimic TF code
					wj = float(B) - torch.sum(yj)
					weights.append(wj)

			scores = torch.stack(scores)  # (K,)
			if self.mode == "weighted":
				weights = torch.stack(weights) + 1e-12
				final_score = torch.sum(scores * weights) / torch.sum(weights)
			else:
				final_score = torch.mean(scores)

			loss = -final_score
			if self.add_constant:
				loss = loss + 1.0
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
		sol_score="tss", 
		sol_distribution="uniform", 
		sol_mode="average",
		sol_add_constant=False,
		**kwargs
	):
		super().__init__(*args, **kwargs)
		self.class_weights = class_weights
		self.multilabel = multilabel
		self.loss_type = loss_type
		self.focal_gamma = focal_gamma
		self.focal_alpha = focal_alpha
		self.sample_weights = sample_weights
		self.sol_score= sol_score
		self.sol_distribution= sol_distribution
		self.sol_mode= sol_mode
		self.sol_add_constant= sol_add_constant

		# - Build the loss criterion
		if self.multilabel:
			if self.loss_type == "ce":
				pos_w = self.class_weights.to(self.model.device) if self.class_weights is not None else None
				self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
				
			elif self.loss_type == "focal":
				pos_w = self.class_weights.to(self.model.device) if self.class_weights is not None else None
				self.loss_fct = FocalLossMultiLabel(gamma=self.focal_gamma, pos_weight=pos_w, reduction="mean")
			
			else:
				raise ValueError(f"Unknown/unsupported loss_type for multilabel classification: {self.loss_type}")
				
		else:
			if self.loss_type == "ce":
				w = self.class_weights.to(self.model.device) if self.class_weights is not None else None
				self.loss_fct = torch.nn.CrossEntropyLoss(weight=w)
		
			elif self.loss_type == "focal":
				alpha = self.focal_alpha
				if isinstance(alpha, torch.Tensor):
					alpha = alpha.to(self.model.device)
				self.loss_fct = FocalLossMultiClass(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
				
			elif self.loss_type == "sol":
				# Note: from_logits=True; multiclass handled automatically
				self.loss_fct = ScoreOrientedLoss(
					score=self.sol_score,
					distribution=self.sol_distribution,
					mu=0.5, delta=0.1,       # ignored for uniform
					mode=self.sol_mode,
					from_logits=True,
					add_constant=self.sol_add_constant,  # usually False (pure -TSS)
				)
            
			else:
				raise ValueError(f"Unknown loss_type: {self.loss_type}")

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
			## SOL expects:
			#  - binary: logits shape (B,) or (B,1), labels (B,)
			#  - multiclass: logits (B,K), labels (B,)
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
		
