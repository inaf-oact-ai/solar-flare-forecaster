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
import gc
from contextlib import nullcontext

# - TORCH
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler

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

# - MOIRAI
try:
	#from uni2ts.data.loader import Collate
	from uni2ts.data.loader import PackCollate as _Collate
except Exception:
	logger.warning("Cannot import uni2ts collate method (not an error if using images/videos) ...")


##########################################
##    DATA COLLATORS
##########################################
class ImgDataCollator:
	def __init__(self, image_processor=None, do_resize=True, do_normalize=True, do_rescale=True):
		self.processor = image_processor
		self.do_resize = do_resize
		self.do_normalize = do_normalize
		self.do_rescale= do_rescale
	
	def __call__(self, batch):
		
		# - Collect batch items
		images, labels = [], []
		
		for item in batch:
			if isinstance(item, dict):
				img = item.get("pixel_values", item.get("image"))
				lab = item.get("labels", item.get("label"))
			else:  # tuple/list
				if not item or item[0] is None:
					continue
				img, lab = item[0], item[1]

			if img is None:
				continue
			
			images.append(img)
			labels.append(lab)
	
		if len(images) == 0:
			# Edge case: all items were None — return empty batch tensors
			return {
				"pixel_values": torch.empty(0),
				"labels": torch.empty(0, dtype=torch.long)
			}
	
		# - Set pixel values ---
		if self.processor is not None:
			# - Pass the raw list to the processor (PIL / np / torch supported)
			proc_out = self.processor(
				images,
				return_tensors="pt",
				do_resize=self.do_resize,       # set False if already resized
				do_normalize=self.do_normalize, # set False if already normalized
				do_rescale=self.do_rescale,     # set False if already rescaled
			)
			pixel_values = proc_out["pixel_values"]
			
		else:
			# Assume items are already tensors; ensure [B, C, H, W]
			images = [img if isinstance(img, torch.Tensor) else torch.as_tensor(img) for img in images] 
			pixel_values = torch.stack(images, dim=0)
	
		# - Set labels
		labels= torch.stack(labels)
	
		return {"pixel_values": pixel_values, "labels": labels}	
		

class VideoDataCollator:
	def __init__(self, image_processor=None, do_resize=True, do_normalize=True, do_rescale=True):
		self.processor = image_processor
		self.do_resize = do_resize
		self.do_normalize = do_normalize
		self.do_rescale= do_rescale

	@staticmethod
	def _to_bcthw(x: torch.Tensor) -> torch.Tensor:
		""" Ensure x is [B, C, T, H, W]. Accepts [B, C, T, H, W] or [B, T, C, H, W]."""
		if x.ndim != 5:
			raise ValueError(f"pixel_values must be 5D, got shape {tuple(x.shape)}")
		B, D1, D2, H, W = x.shape
		# Heuristic: channels is almost always 1 or 3
		if D1 in (1, 3):           # [B, C, T, H, W]
			return x
		if D2 in (1, 3):           # [B, T, C, H, W] -> [B, C, T, H, W]
			return x.permute(0, 2, 1, 3, 4)
		# Fallback: assume already [B, C, T, H, W]
		return x

	def __call__(self, batch):
	
		# - Collect batch items
		videos, labels = [], []
		
		for item in batch:
			# - item[0] is a list of T tensor of Shape: [C,H,W])
			if not item or item[0] is None:
				continue
			frames, lab = item[0], item[1]

			if frames is None:
				continue
			
			videos.append(frames)
			labels.append(lab)
			
		if len(videos) == 0:
			raise ValueError("Empty batch after filtering invalid items.")	
			
		# - Set pixel values ---
		if self.processor is not None:
			# - NB: This works only with list of list of [C,H,W]
			proc_out = self.processor(
				videos,                      # list of length B; each item is list of T HWC frames
				return_tensors="pt",
				do_resize=self.do_resize,        # set False if you already resized
				do_normalize=self.do_normalize,  # set False if you already normalized
				do_rescale=self.do_rescale,      # set False if already rescaled
			)
			pixel_values = proc_out["pixel_values"].float()  # Shape: [B,T,C,H,W]
			
		else:
			# - Convert 2D list to tensor of Shape: [B,T,C,H,W]
			vids_tchw = [torch.stack(item, dim=0) for item in videos]
			pixel_values = torch.stack(vids_tchw, dim=0).float()

		#print("pixel_values")
		#print(pixel_values.shape)

		# - VideoMAE model require a Tensor of Shape: [B,C,T,H,W]
		###pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # Tensor of Shape: [B,C,T,H,W]
		#pixel_values = self._to_bcthw(pixel_values)
			
		# - Set labels
		labels= torch.stack(labels)
		
		# - Check if any NaN in pixel_values
		if torch.isnan(pixel_values).any():
			print("⚠️ NaN values detected in batch tensor!")

		if torch.isinf(pixel_values).any():
			print("⚠️ Inf values detected in batch tensor!")
		
		return {"pixel_values": pixel_values, "labels": labels}	
		


class TSDataCollator:
	#def __init__(self):
		
	def __call__(self, batch):	
		
		# - Collect batch items
		ts_list, label_list = [], []
		
		for item in batch:
			if isinstance(item, dict):
				ts = item.get("input", item.get("input"))
				lab = item.get("labels", item.get("label"))
			else:  # tuple/list
				if not item or item[0] is None:
					continue
				ts, lab = item[0], item[1]

			if ts is None:
				continue
			
			ts_list.append(ts)
			label_list.append(lab)
	
		if len(ts_list) == 0:
			# Edge case: all items were None — return empty batch tensors
			return {
				"input": torch.empty(0),
				"labels": torch.empty(0, dtype=torch.long)
			}
			
		# - Apply here model processor logic (if any processor)
		# ...
		# ...	
			
		# - Set features
		features = torch.stack(ts_list, dim=0)
		
		# - Set labels
		labels= torch.stack(label_list)
		
		# - Check if any NaN in pixel_values
		if torch.isnan(features).any():
			print("⚠️ NaN values detected in batch tensor!")

		if torch.isinf(features).any():
			print("⚠️ Inf values detected in batch tensor!")
		
		return {"input": features, "labels": labels}	
		
	
class Uni2TSBatchCollator:
	"""
		Wraps Uni2TS Collate to produce the tokenized batch for Moirai{1,2}.forward(...)
		from simple samples: {"input": [T,C], "labels": int}.
	"""
	def __init__(self, context_length: int, patch_size: int):
		# Collate args: target_field name must match what backbone expects ("target")
		# max_length ≈ context_length (will be patched/packed inside)
		self._collate = _Collate(
			target_field="target",
			seq_fields=("target", "observed_mask"),
			max_length=context_length,
			#patch_size=patch_size,
			# other defaults are fine for classification (no forecast region)
		)

	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
		# Prepare "raw" samples in Uni2TS expected pre-collate form
		raw_batch = []
		labels = []
		
		for it in features:
			x, y = split_feature(it)
			if x is None:
				continue
			T, C = x.shape
			obs = np.ones((T, C), dtype=np.bool_)
			raw_batch.append({"target": x, "observed_mask": obs})
			labels.append(y)

		if not raw_batch:  # edge case: all invalid/None
			return {
				"target": torch.empty(0), "observed_mask": torch.empty(0, dtype=torch.bool),
				"sample_id": torch.empty(0, dtype=torch.long), "time_id": torch.empty(0, dtype=torch.long),
				"variate_id": torch.empty(0, dtype=torch.long), "prediction_mask": torch.empty(0, dtype=torch.bool),
				"labels": torch.empty(0, dtype=torch.long)
			}

		packed = self._collate(raw_batch)  # numpy arrays
		packed["labels"] = np.asarray(labels, dtype=np.int64)
		
		
		# target/observed_mask should already be [B, L, P] (or [B, L] for observed_mask in some builds)
		target = to_np(packed["target"])                 # [B, L, P]
		obs    = to_np(packed["observed_mask"])          # [B, L, P] (or [B, L] in some commits)
		B, L = target.shape[0], target.shape[1]
		#print("target.shape")
		#print(target.shape)
		#print("obs.shape")
		#print(obs.shape)		
		

		# Ensure required IDs/masks exist; synthesize if missing.
		
		# sample_id should be [B, L] in your build — don't flatten it
		if "sample_id" not in packed:
			# Fill with row indices (0..B-1)
			sample_id = np.tile(np.arange(B, dtype=np.int64)[:, None], (1, L))
			packed["sample_id"] = sample_id
		else:
			packed["sample_id"] = to_np(packed["sample_id"], np.int64)
			if packed["sample_id"].ndim == 1:
				# expand to [B,L] if needed (rare)
				packed["sample_id"] = np.tile(packed["sample_id"][None, :], (B, 1))

		#print("packed['sample_id'].shape")
		#print(packed["sample_id"].shape)

		# time_id: per-timestep index [B, L]
		if "time_id" not in packed:
			time_id = np.tile(np.arange(L, dtype=np.int64)[None, :], (B, 1))
			packed["time_id"] = time_id
		else:
			packed["time_id"] = to_np(packed["time_id"], np.int64)
    
		#print("packed['time_id'].shape")
		#print(packed["time_id"].shape)
		
		# variate_id: zeros [B, L] (you can encode channel IDs later)
		if "variate_id" not in packed:
			packed["variate_id"] = np.zeros((B, L), dtype=np.int64)
		else:
			packed["variate_id"] = to_np(packed["variate_id"], np.int64)
		
		#print("packed['variate_id'].shape")
		#print(packed["variate_id"].shape)

		# prediction_mask: all False [B, L] for classification
		if "prediction_mask" not in packed:
			packed["prediction_mask"] = np.zeros((B, L), dtype=np.bool_)
		else:
			packed["prediction_mask"] = to_np(packed["prediction_mask"], np.bool_)

		#print("packed['prediction_mask'].shape")
		#print(packed["prediction_mask"].shape)

		# Finally convert everything numpy→torch with correct dtypes
		out = {}
		for k, v in packed.items():
			if isinstance(v, np.ndarray):
				if v.dtype == np.bool_:
					out[k] = torch.from_numpy(v)                        # bool
				elif np.issubdtype(v.dtype, np.integer):
					out[k] = torch.from_numpy(v.astype(np.int64))       # int64
				else:
					out[k] = torch.from_numpy(v.astype(np.float32))     # float32
			elif isinstance(v, torch.Tensor):
				out[k] = v
			else:
				# labels etc. if still python scalars/lists
				out[k] = torch.tensor(v)
		
		return out
	
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
		score_fn: str = "tss",
		distribution: str = "uniform",
		mu: Union[float, List[float]] = 0.5,
		delta: Union[float, List[float]] = 0.1,
		mode: str = "average",
		#from_logits: bool = True,
		add_constant: bool = False,
	):
		super().__init__()
		
		_VALID_SCORE_FN= ["accuracy", "precision", "recall", "specificity", "f1", "tss", "csi", "hss1", "hss2"]
		
		assert score_fn in _VALID_SCORE_FN, f"Unknown score: {score_fn}"
		assert distribution in ("uniform", "cosine")
		assert mode in ("average", "weighted")
		self.score_fn = score_fn
		self.distribution = distribution
		self.mu = mu
		self.delta = delta
		self.mode = mode
		#self.from_logits = from_logits
		self.add_constant = add_constant


	def _F_uniform(self, p: torch.Tensor) -> torch.Tensor:
		# uniform prior over threshold ⇒ F(p) = p
		return p

	def _F_cosine(self, p: torch.Tensor, mu: float, delta: float) -> torch.Tensor:
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

	def _apply_distribution(self, p: torch.Tensor, j: Optional[int] = None) -> torch.Tensor:
		""" Select distribution """
		if self.distribution == "uniform":
			return self._F_uniform(p)
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
			return self._F_cosine(p, mu, delta)

	def _expected_confusion(self, y_true, p, j: Optional[int] = None):
		"""
			y_true: (B,) in {0,1}  float or long
			p     : (B,) in [0,1]  probability for positive class
			Returns scalars TP,TN,FP,FN with grad.
		"""
		y = y_true.float()
		# F_unif(p) = p
		# Fp = p
		Fp = self._apply_distribution(p, j=j).clamp(0.0, 1.0)
		
		TN = torch.sum((1.0 - y) * (1.0 - Fp))
		TP = torch.sum(y * Fp)
		FP = torch.sum((1.0 - y) * Fp)
		FN = torch.sum(y * (1.0 - Fp))
		
		return TN, FP, FN, TP

	def _compute_score_from_confusion(self, TN, FP, FN, TP, which):
		""" Compute score from confusion values """
		# all tensors (scalar) with grad
		eps = 1e-12
		which = which.lower()
    
		if which == 'tss':
			# recall + specificity - 1
			#rec = TP / torch.nan_to_num(TP + FN, nan=0.0)      # TP / (TP+FN)
			#spe = TN / torch.nan_to_num(TN + FP, nan=0.0)      # TN / (TN+FP)
			
			rec_den = TP + FN
			spe_den = TN + FP
			rec = torch.where(rec_den > 0, TP / (rec_den + eps), torch.zeros_like(TP))
			spe = torch.where(spe_den > 0, TN / (spe_den + eps), torch.zeros_like(TN))
			return rec + spe - 1.0
			
		elif which == 'accuracy':
			#return (TP + TN) / torch.nan_to_num(TP + TN + FP + FN, nan=0.0)
			den = TP + TN + FP + FN
			return torch.where(den > 0, (TP + TN) / (den + eps), torch.zeros_like(den))

		elif which == 'precision':
			#return TP / torch.nan_to_num(TP + FP, nan=0.0)
			den = TP + FP
			return torch.where(den > 0, TP / (den + eps), torch.zeros_like(den))

		elif which == 'recall':
			#return TP / torch.nan_to_num(TP + FN, nan=0.0)
			den = TP + FN
			return torch.where(den > 0, TP / (den + eps), torch.zeros_like(den))
			
		elif which == 'specificity':
			#return TN / torch.nan_to_num(TN + FP, nan=0.0)
			den = TN + FP
			return torch.where(den > 0, TN / (den + eps), torch.zeros_like(den))
			
		elif which == 'f1':
			prec = TP / torch.nan_to_num(TP + FP, nan=0.0)
			rec  = TP / torch.nan_to_num(TP + FN, nan=0.0)
			return 2 * (prec * rec) / torch.nan_to_num(prec + rec, nan=0.0)
			
		elif which == 'csi':
			return TP / torch.nan_to_num(TP + FP + FN, nan=0.0)
		elif which == 'hss1':
			return (TP - FP) / torch.nan_to_num(TP + FN, nan=0.0)
		elif which == 'hss2':
			num = 2 * (TP * TN - FP * FN)
			den = (TP + FN) * (FN + TN) + (TP + FP) * (TN + FP)
			return num / torch.nan_to_num(den, nan=0.0)
		else:
			raise ValueError(f"Unknown SOL score: {which}")


	def _compute_binary_score(self, logits, labels):
		""" Compute binary class loss """

		# binary: logits (B,1) or (B,)
		if logits.ndim == 2 and logits.shape[-1] == 1:
			logits_bin = logits.squeeze(-1)
		else:
			logits_bin = logits
			
		prob_pos = torch.sigmoid(logits_bin)                 # (B,)
		y_bin    = labels.float().view(-1)                   # (B,)
		TN, FP, FN, TP = self._expected_confusion(y_bin, prob_pos, j=None)
		score = self._compute_score_from_confusion(TN, FP, FN, TP, which=self.score_fn)
		
		return score
		

	def _compute_multiclass_score(self, logits, labels):
		""" Compute multiclass loss """
		
		C = logits.shape[-1]
		
		# multiclass one-vs-rest on softmax probabilities
		probs = torch.softmax(logits, dim=-1)                # (B,C)
		y_idx = labels.view(-1).long()                       # (B,)
		
		#print("probs")
		#print(probs)
		#print(probs.shape)
		#print("y_idx")
		#print(y_idx)
		#print(y_idx.shape)
    
		# build one-hot without breaking grad path
		y_onehot = torch.zeros_like(probs).scatter_(1, y_idx.unsqueeze(1), 1.0)  # (B,C)
		
		#print("y_onehot")
		#print(y_onehot)
		#print(y_onehot.shape)

		per_class_scores = []
		per_class_weights = []  # for 'weighted' mode: weight by #negatives like the TF ref
		
		for j in range(C):
			p_j = probs[:, j]               # (B,)
			y_j = y_onehot[:, j]            # (B,)
			#print(f"p_{j}")
			#print(p_j)
			#print(p_j.shape)
			#print(f"y_{j}")
			#print(y_j)
			#print(y_j.shape)
			
			TN, FP, FN, TP = self._expected_confusion(y_j, p_j, j=j)
			s_j = self._compute_score_from_confusion(TN, FP, FN, TP, which=self.score_fn)
			per_class_scores.append(s_j)
			
			#print("TN")
			#print(TN)
			#print("FP")
			#print(FP)
			#print("FN")
			#print(FN)
			#print("TP")
			#print(TP)
			#print(f"s_{j}")
			#print(s_j)

			# weight by #negatives in batch (like original SOL 'weighted' option)
			n_neg = torch.clamp((y_j.shape[0] - y_j.sum()), min=1.0)
			per_class_weights.append(n_neg)

		scores = torch.stack(per_class_scores)               # (C,)
		if self.mode.lower() == 'weighted':
			w = torch.stack(per_class_weights)               # (C,)
			score = (scores * w).sum() / w.sum()
		else:
			score = scores.mean()
			
		#print("score")
		#print(score)
			
		return score

	def forward(self, logits, labels):
	
		# - Compute score
		C = logits.shape[-1] if logits.ndim == 2 else 1
		
		if C == 1: # binary
			#print(f"Computing binary score (C={C}, logits.ndim={logits.ndim}, logits.shape[-1]={logits.shape[-1]}) ...")
			score= self._compute_binary_score(logits, labels)
		else:
			#print(f"Computing multiclass score (C={C}, logits.ndim={logits.ndim}, logits.shape[-1]={logits.shape[-1]})) ...")
			score= self._compute_multiclass_score(logits, labels)
			
		# - Compute final loss
		if self.add_constant: # TSS=[-1,1] --> LOSS=-TSS=[-1,1] --> LOSS=[0,2] 
			loss_sol= -score + 1.0
		else:
			loss_sol= -score
			
		#print("loss_sol")
		#print(loss_sol)

		return loss_sol
		

##########################################
##    DATA SAMPLERS
##########################################
class DistributedWeightedRandomSampler(Sampler):
	def __init__(
		self, 
		weights, 
		num_samples=None,
		replacement=True,
		num_replicas=None, 
		rank=None, 
		seed=0
	):
		self.weights = torch.as_tensor(weights, dtype=torch.float)
		self.N = len(self.weights)
		if num_replicas is None:
			if dist.is_available() and dist.is_initialized():
				num_replicas = dist.get_world_size()
			else:
				num_replicas = 1
		if rank is None:
			if dist.is_available() and dist.is_initialized():
				rank = dist.get_rank()
			else:
				rank = 0
		self.num_replicas = num_replicas
		self.rank = rank
		self.replacement = replacement
		self.seed = seed
		self.epoch = 0

		# build initial shard mapping like DistributedSampler
		self.num_samples_per_replica = int(math.ceil(self.N / self.num_replicas))
		self.total_size = self.num_samples_per_replica * self.num_replicas
		self.num_samples = num_samples or self.num_samples_per_replica
		self._build_local_indices()

	def _build_local_indices(self):
		# deterministic shuffle per epoch like DistributedSampler
		g = torch.Generator()
		g.manual_seed(self.seed + self.epoch)
		perm = torch.randperm(self.N, generator=g)

		# pad then shard
		if self.total_size > self.N:
			padding = perm[: self.total_size - self.N]
			perm = torch.cat([perm, padding], dim=0)
        
		# contiguous chunk for this rank
		start = self.rank * self.num_samples_per_replica
		end = start + self.num_samples_per_replica
		self.local_indices = perm[start:end]
		self.local_weights = self.weights[self.local_indices]

	def __iter__(self):
		# independent draw on local shard
		g = torch.Generator()
		g.manual_seed(self.seed + self.epoch + 12345 + self.rank)
		if self.replacement:
			picks = torch.multinomial(self.local_weights, self.num_samples, replacement=True, generator=g)
		else:
			k = min(self.num_samples, self.local_indices.numel())
			picks = torch.multinomial(self.local_weights, k, replacement=False, generator=g)
		yield from self.local_indices[picks].tolist()

	def __len__(self):
		return self.num_samples

	def set_epoch(self, epoch: int):
		self.epoch = epoch
		self._build_local_indices()

##########################################
##    CUSTOM TRAINER
##########################################
class CustomTrainer(Trainer):
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
		ordinal=False,
		ordinal_pos_weights=None,
		compute_train_metrics=False,
		binary_pos_weights=None, 
		logitout_size=4,
		verbose=False,
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
		self.ordinal= ordinal
		self.ordinal_pos_weights= ordinal_pos_weights
		self.verbose= verbose
		self.compute_train_metrics = compute_train_metrics
		self.binary_pos_weights= binary_pos_weights
		self.logitout_size= logitout_size
		self._reset_train_buffers()
		
		self.is_binary_single_logit = (
			not self.multilabel and
			not self.ordinal and
			self.logitout_size==1
			#getattr(self.model.config, "num_labels", None) == 1
		)
		
		#print("self.model.config")
		#print(self.model.config)
		#print("self.is_binary_single_logit")
		#print(self.is_binary_single_logit)
		#print(getattr(self.model.config, "num_labels", None))
		#print(self.logitout_size)
		
		#dev= self.model.device
		dev = next(self.model.parameters()).device

		# - Build the loss criterion
		if self.multilabel:
			#########################
			##  MULTI-LABEL
			#########################
			if self.loss_type == "ce":
				pos_w = self.class_weights.to(dev) if self.class_weights is not None else None
				self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
				
			elif self.loss_type == "focal":
				pos_w = self.class_weights.to(dev) if self.class_weights is not None else None
				self.loss_fct = FocalLossMultiLabel(gamma=self.focal_gamma, pos_weight=pos_w, reduction="mean")
			
			else:
				raise ValueError(f"Unknown/unsupported loss_type for multilabel classification: {self.loss_type}")
			
		elif self.ordinal:
			#########################
			##  ORDINAL
			#########################
			pos_w = self.ordinal_pos_weights.to(dev) if self.ordinal_pos_weights is not None else None
			
			if self.loss_type == "ce":
				# BCEWithLogitsLoss with per-threshold pos_weight
				self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
			
			elif self.loss_type == "focal":
				## --> ADD IMPLEMENTATION HERE
				#raise ValueError(f"loss_type {self.loss_type} not yet implemented")
				# Multi-label focal loss variant, again with per-threshold pos_weight
				self.loss_fct = FocalLossMultiLabel(
					gamma=self.focal_gamma,
					pos_weight=pos_w,
					reduction="mean",
				)
			
			elif self.loss_type == "sol":
				raise ValueError(f"loss_type {self.loss_type} not supported for ordinal model")
				
			else:
				raise ValueError(f"Unknown/unsupported loss_type for ordinal classification: {self.loss_type}")
			
		else:
			#########################
			##  SINGLE-LABEL
			#########################
			if self.loss_type == "ce":
				#w = self.class_weights.to(dev) if self.class_weights is not None else None
				#self.loss_fct = torch.nn.CrossEntropyLoss(weight=w)
		
				if self.is_binary_single_logit:
					# BCE for single-logit binary
					pos_w = (self.binary_pos_weights.to(dev) if self.binary_pos_weights is not None else None)
					self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
				else:
					w = self.class_weights.to(dev) if self.class_weights is not None else None
					self.loss_fct = torch.nn.CrossEntropyLoss(weight=w)	
				
		
			elif self.loss_type == "focal":
				#alpha = self.focal_alpha
				#if isinstance(alpha, torch.Tensor):
				#	alpha = alpha.to(dev)
				#self.loss_fct = FocalLossMultiClass(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
				
				if self.is_binary_single_logit:
					# Use BCE-style focal via multilabel focal (C=1)
					pos_w = (self.binary_pos_weights.to(dev) if self.binary_pos_weights is not None else None)
					self.loss_fct = FocalLossMultiLabel(
						gamma=self.focal_gamma,
						pos_weight=pos_w,      # class tilt for positives
						reduction="mean",
					)
				else:
					alpha = self.focal_alpha
					if isinstance(alpha, torch.Tensor):
						alpha = alpha.to(dev)
					self.loss_fct = FocalLossMultiClass(alpha=alpha, gamma=self.focal_gamma, reduction="mean")
				
			elif self.loss_type == "sol":
				# Note: from_logits=True; multiclass handled automatically
				self.loss_fct = ScoreOrientedLoss(
					score_fn=self.sol_score,
					distribution=self.sol_distribution,
					mu=0.5, 
					delta=0.1,       # ignored for uniform
					mode=self.sol_mode,
					#from_logits=True,
					add_constant=self.sol_add_constant,  # usually False (pure -TSS)
				)
            
			else:
				raise ValueError(f"Unknown loss_type: {self.loss_type}")

	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		""" Override trainer compute_loss function """
		
		# - Retrieve features & labels
		if "target" in inputs and "sample_id" in inputs:     # Uni2TS packed batch
			#print("--> debug inputs")
			#dbg(inputs)	 # debug printout
			outputs = model(**inputs)
		else:
			#features = inputs.get("pixel_values") or inputs.get("input")
			#outputs = model(features)  # legacy path
			
			features = inputs.get("pixel_values", None)
			if features is not None:
				# image/video (and hybrid image→TS) path
				outputs = model(pixel_values=features)
			else:
				features = inputs.get("input", None)
				if features is None:
					raise KeyError("Neither 'pixel_values' nor 'input' found in inputs.")
				# plain TS models that accept raw 'input'
				outputs = model(features)
				
			if torch.is_tensor(features):
				if torch.isnan(features).any():
					print("⚠️ NaN values detected in batch features tensor!")
				if torch.isinf(features).any():
					print("⚠️ Inf values detected in batch features tensor!")
    
		labels = inputs.get("labels")
		logits = outputs.logits
    
		#features = inputs.get("pixel_values", None)
		#if features is None:
		#	features = inputs.get("input", None)  # <-- time series
    
    #outputs = model(features)    
		#labels = inputs.get("labels")
		#logits = outputs.logits
		
		if torch.isnan(labels).any() or torch.isinf(labels).any():
			print("⚠️ NaN values detected in batch label tensor!")
				
		if torch.isnan(logits).any() or torch.isinf(logits).any():
			print("⚠️ NaN values detected in logits tensor!")	
		
		# ---- shape fix for single-logit binary + BCE/focal ----
		if self.is_binary_single_logit and self.loss_type in ("ce", "focal"):
			# logits: (B,1), labels: (B,) -> (B,1)
			labels = labels.float().view(-1, 1)

		if self.multilabel or (self.is_binary_single_logit and self.loss_type in ("ce","focal")):
			# BCE-style losses expect float targets
			labels = labels.float()
		
		loss = self.loss_fct(logits, labels)
		
		#if self.multilabel:
		#	labels = labels.float()
		#	loss = self.loss_fct(logits, labels)
		#else:
		#	## SOL expects:
		#	#  - binary: logits shape (B,) or (B,1), labels (B,)
		#	#  - multiclass: logits (B,K), labels (B,)
		#	loss = self.loss_fct(logits, labels)
			
		if self.verbose:
			print("logits")
			print(logits)
			print("labels")
			print(labels)
			print("loss")
			print(loss)
			
		# - Update train metric variables (if required to be computed)
		if self.compute_train_metrics:
			if logits is not None and labels is not None:
				#print("Updating train metric counters ...")
				# Detach now to avoid holding graph; keep on device for cheap gather
				self._train_logits.append(logits.detach())
				self._train_labels.append(labels.detach())


		return (loss, outputs) if return_outputs else loss

	
	def _get_train_sampler(self, *args, **kwargs):
		# When we build our own weighted sampler, do not let HF create another one.
		if self.sample_weights is not None:
			return None
		return super()._get_train_sampler(*args, **kwargs)
		

	def get_train_dataloader(self):
		""" Get train dataloader with resampling """
		
		sw = self.sample_weights

		# --- No weights: delegate to base Trainer (this is already distributed) ---
		if sw is None:
			logger.info("No sample weights given, using standard train dataloader ...")
			dl = super().get_train_dataloader()
			if self.args.local_rank in (-1, 0):
				print("Sampler:", type(dl.sampler).__name__)
				try:
					print("Replicas/rank:", dl.sampler.num_replicas, dl.sampler.rank)
				except AttributeError:
					pass
			return dl
		
		# --- Defensive checks on weights ---
		if len(sw) != len(self.train_dataset):
			raise ValueError(f"sample_weights length ({len(sw)}) != train_dataset length ({len(self.train_dataset)}). Make sure weights are computed AFTER any filtering/subsetting.")
    
		import torch as _torch
		sw = _torch.as_tensor(sw, dtype=_torch.float)
				
		# --- DDP path: use your shard-aware sampler ---
		if (getattr(self.args, "world_size", 1) or 1) > 1:
			logger.info("DDP detected, using shard-aware sampler ...")
			# IMPORTANT: class should implement set_epoch(epoch)
			sampler = DistributedWeightedRandomSampler(
				weights=sw,
				# Let the sampler compute per-replica length internally or pass explicitly:
				# num_samples=None,
				replacement=True,
				num_replicas=self.args.world_size,
				rank=self.args.process_index,
				seed=self.args.seed,
			)
			dl = DataLoader(
				self.train_dataset,
				batch_size=self.args.train_batch_size,
				sampler=sampler,
				collate_fn=self.data_collator,
				num_workers=self.args.dataloader_num_workers,
				pin_memory=self.args.dataloader_pin_memory,
				drop_last=self.args.dataloader_drop_last,
				persistent_workers=getattr(self.args, "dataloader_persistent_workers", False) and self.args.dataloader_num_workers > 0,
			)
			return dl
        
		
		# --- Single-process path: standard weighted sampler ---
		logger.info("Using weighted random sampler ...")
		sampler = WeightedRandomSampler(
			weights=sw,
			num_samples=len(sw),
			replacement=True,
			generator=torch.Generator().manual_seed(self.args.seed),
		)
		return DataLoader(
			self.train_dataset,
			batch_size=self.args.train_batch_size,
			sampler=sampler,
			collate_fn=self.data_collator,
			num_workers=self.args.dataloader_num_workers,
			pin_memory=self.args.dataloader_pin_memory,
			drop_last=self.args.dataloader_drop_last,
			persistent_workers=getattr(self.args, "dataloader_persistent_workers", False) and self.args.dataloader_num_workers > 0,
		)
		
	
	def _reset_train_buffers(self):
		self._train_logits = []
		self._train_labels = []
		
	def _extract_labels(self, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
		# common label keys: "labels" or "label"
		for k in ("labels", "label"):
			if k in inputs and inputs[k] is not None:
				return inputs[k]
		return None
		
	def _gather_for_metrics(self, t: torch.Tensor) -> torch.Tensor:
		"""
			Gather a tensor across processes for metrics. Uses Accelerate utilities
			when available; falls back to identity in single-process.
		"""
		if hasattr(self, "accelerator") and self.accelerator is not None:
			# Handles uneven batch sizes & different shapes across steps
			return self.accelerator.gather_for_metrics(t)
		
		# Single-process fallback
		return t
		
	def _compute_and_log_train_metrics(self):
		""" Compute train metrics and log them """
		
		if not self.compute_train_metrics:
			return
			
		if self.compute_metrics is None:
			print("compute_metrics is None, no train metrics computed ...")
			return

		if len(self._train_logits) == 0 or len(self._train_labels) == 0:
			print("WARN: train logits/labels empty, no train metrics computed ...")
			return  # nothing collected (e.g., gradient_accum_only run)

		# Concatenate per-step tensors
		logits = torch.cat(self._train_logits, dim=0)
		labels = torch.cat(self._train_labels, dim=0)

		# Gather across processes for correct global metrics
		logits = self._gather_for_metrics(logits)
		labels = self._gather_for_metrics(labels)

		# Move to CPU numpy for compute_metrics
		preds_np = logits.cpu().numpy()
		labels_np = labels.cpu().numpy()

		# Use the same compute_metrics you pass to Trainer
		ep = EvalPrediction(predictions=preds_np, label_ids=labels_np)
		metrics = self.compute_metrics(ep)  # user-defined

		# Prefix keys to distinguish from eval metrics
		#metrics = {f"train/{k}": v for k, v in metrics.items()}
		
		print("--> train_metrics")
		print(metrics)

		# Log via Trainer's logger (goes to W&B if report_to includes "wandb")
		self.log(metrics)
		
	#def _compute_pos_weight_from_ce_weights(self):
	#	if self.class_weights is None or self.class_weights.numel() < 2:
	#		return None
	#	w_neg, w_pos = self.class_weights[0].to(self.model.device), self.class_weights[1].to(self.model.device)
	#	return (w_pos / (w_neg + 1e-12))	
		
	
class CustomTrainerTS(CustomTrainer):
	"""
		TS-only variant: override prediction_step so HF can collect preds/labels.
		This leaves the base CustomTrainer behavior unchanged for image/video.
	"""
	
	#def training_step(self, model, inputs):
	#	# Enable only for the first few steps to reduce overhead
	#	enable = (self.state.global_step < 5)
	#	ctx = torch.autograd.detect_anomaly() if enable else nullcontext()
	#	with ctx:
	#		return super().training_step(model, inputs)
	
	def prediction_step(
		self,
		model: torch.nn.Module,
		inputs: dict,
		prediction_loss_only: bool,
		ignore_keys: Optional[list] = None,
	) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
		model.eval()

		# Labels (required for metrics)
		labels = inputs.get("labels", None)
		has_labels = labels is not None

		# Forward pass with Moirai’s packed fields
		with torch.no_grad():
			outputs = model(
				target=inputs["target"],                  # [B,L,P]
				observed_mask=inputs["observed_mask"],   # [B,L,P]
				sample_id=inputs["sample_id"],           # [B,L]
				time_id=inputs["time_id"],               # [B,L]
				variate_id=inputs["variate_id"],         # [B,L]
				prediction_mask=inputs["prediction_mask"]# [B,L]
			)

		# Your model returns SequenceClassifierOutput(logits=...)
		logits = outputs.logits

		# Return shape expected by HF: (loss, logits, labels). We don’t compute loss here
    # because your CustomTrainer.compute_loss already owns loss logic.
		loss = None
		if prediction_loss_only:
			return (loss, None, None)

		return (loss, logits, labels)	
		
class TrainMetricsCallback(TrainerCallback):
	"""
		Hooks that DO get called: on_epoch_begin / on_epoch_end.
		Works with any Trainer, but expects the methods we defined on CustomTrainer.
	"""
	def __init__(self, trainer: CustomTrainer):
		super().__init__()
		self.trainer = trainer

	def on_epoch_begin(self, args, state, control, **kwargs):
		if getattr(self.trainer, "compute_train_metrics", False):
			#print("Resetting train metrics buffers ...")
			self.trainer._reset_train_buffers()

	def on_epoch_end(self, args, state, control, **kwargs):
		if getattr(self.trainer, "compute_train_metrics", False):
			#print("Computing and logging train metrics ...")
			self.trainer._compute_and_log_train_metrics()
		# No change to control flow
		return control

class CudaGCCallback(TrainerCallback):
	"""
		Light-weight CUDA cache purger to reduce fragmentation.
		Mirrors TrainMetricsCallback style: holds `trainer`, returns `control`.
	"""
	def __init__(self, trainer, verbose: bool = False, every_n_logs: int | None = None):
		super().__init__()
		self.trainer = trainer
		self.verbose = verbose
		self.every_n_logs = every_n_logs

	def _purge(self, tag: str):
		# Per-rank purge; safe under DDP (no barriers here!)
		if torch.cuda.is_available():
			try:
				torch.cuda.synchronize()
			except Exception:
				pass
			torch.cuda.empty_cache()

		gc.collect()
		if self.verbose and torch.cuda.is_available():
			# quick per-device snapshot
			for i in range(torch.cuda.device_count()):
				a = torch.cuda.memory_allocated(i) / 1e9
				r = torch.cuda.memory_reserved(i) / 1e9
				print(f"[{tag}] GPU{i}: alloc={a:.2f}G reserved={r:.2f}G")

	def on_epoch_begin(self, args, state, control, **kwargs):
		# start each epoch with a clean cache
		self._purge("epoch_begin")
		return control

	def on_evaluate(self, args, state, control, **kwargs):
		# called after evaluation finishes
		self._purge("after_evaluate")
		return control

	def on_save(self, args, state, control, **kwargs):
		# called after a checkpoint is saved
		self._purge("after_save")
		return control

	def on_epoch_end(self, args, state, control, **kwargs):
		# after your TrainMetricsCallback computes/logs metrics, purge again
		self._purge("epoch_end")
		return control

	def on_log(self, args, state, control, **kwargs):
		if self.every_n_logs and state.global_step and state.global_step % self.every_n_logs == 0:
			self._purge("on_log")
		return control
		
