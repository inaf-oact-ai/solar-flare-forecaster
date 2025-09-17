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
from typing import Optional

# - TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification
from transformers import ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.configuration_utils import PretrainedConfig

# - MODULE
from sfforecaster import logger

# - Import Moirai2 if available, else Moirai1
try:
	from uni2ts.model.moirai2 import Moirai2Module as _M
except Exception:
	try:
		from uni2ts.model.moirai import MoiraiModule as _M
	except Exception:
		logger.warning("Cannot import Moirai 1/2 (is uni2ts installed?), cannot use time series models!")

###########################################
###     CORAL ORDINAL HEAD MODEL
###########################################
class CoralOrdinalHead(nn.Module):
	"""
		Structural ordinal head for K ordered classes.
		Produces K-1 logits corresponding to [y>=class_1, ..., y>=class_{K-1}].

		For your case K=4 (NONE < C < M < X) -> 3 logits: [>=C, >=M, >=X].
	"""
	def __init__(self, in_features: int, num_classes: int):
		super().__init__()
		assert num_classes >= 2
		self.num_classes = num_classes
		self.num_thresholds = num_classes - 1

		# Single scalar score s(x)
		self.score = nn.Linear(in_features, 1)

		# Unconstrained parameters for thresholds; will be mapped to strictly increasing cutpoints
		self.theta_unconstrained = nn.Parameter(torch.zeros(self.num_thresholds))

		# Optional: small bias to start thresholds roughly ordered
		nn.init.normal_(self.theta_unconstrained, mean=0.0, std=0.02)

	def ordered_cutpoints(self):
		# Softplus ensures positivity, cumsum enforces strict monotonicity
		# You can add an offset if needed; usually not necessary.
		deltas = F.softplus(self.theta_unconstrained)
		return torch.cumsum(deltas, dim=0)  # shape: (K-1,)

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		"""
			features: (B, in_features)
			returns logits: (B, K-1) for cumulative targets [>=C, >=M, >=X]
		"""
		s = self.score(features)                  # (B, 1)
		cut = self.ordered_cutpoints()           # (K-1,)
		logits = s - cut.view(1, -1)             # broadcast to (B, K-1)
		return logits


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
		
		
##############################################
###     TIME-SERIES CLASSIFICATION MODEL
##############################################    
class MoiraiTSConfig(PretrainedConfig):
	model_type = "moirai_ts_classifier"
	def __init__(
		self, 
		num_labels: int = 4, 
		d_model: int = 384, 
		**kwargs
	):
		super().__init__(**kwargs)
		self.num_labels = num_labels
		self.d_model = d_model
        
class MoiraiForSequenceClassification(torch.nn.Module):
	"""
		Loss-agnostic wrapper: returns logits only.
		Compatible with your CustomTrainer that owns the loss.
		Accepts either `input=[B,T,C]` or `pixel_values=[B,T,C]`.
	"""
	def __init__(
		self, 
		pretrained_name: str = "Salesforce/moirai-2.0-R-small",
		num_labels: int = 4, 
		freeze_backbone: bool = False
	):
		super().__init__()
		self.backbone = _M.from_pretrained(pretrained_name)
		
		d_model = getattr(self.backbone, "d_model", 384)
		self.config = MoiraiTSConfig(num_labels=num_labels, d_model=d_model)

		if freeze_backbone:
			for p in self.backbone.parameters():
				p.requires_grad = False

		# - Try to make the backbone emit representations
		for attr in ("return_repr", "output_hidden_states", "return_hidden"):
			if hasattr(self.backbone, attr):
				try: setattr(self.backbone, attr, True)
				except Exception: pass

		self._last_repr = None
		self._hooked = self._register_hook_on_any(["transformer","encoder","backbone","model","net"])

		self.pool = torch.nn.AdaptiveAvgPool1d(1)
		self.head = torch.nn.Sequential(
			torch.nn.Linear(d_model, d_model),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.1),
			torch.nn.Linear(d_model, num_labels),
		)

	def _register_hook_on_any(self, names):
		for name in names:
			mod = getattr(self.backbone, name, None)
			if mod is None: 
				continue
			try:
				def _hook(module, inp, out):
					val = None
					if isinstance(out, (list, tuple)) and len(out) > 0: 
						val = out[0]
					elif isinstance(out, dict):
						val = out.get("reprs", None) or out.get("hidden_states", None) or out.get("x", None)
					else:
						val = out
					if isinstance(val, torch.Tensor):
						self._last_repr = val
				mod.register_forward_hook(_hook)
				return True
			except Exception:
				continue
		return False
		
	def _backbone_forward(self, batch_dict: dict) -> torch.Tensor:
		"""
			batch_dict contains: target, observed_mask, sample_id, time_id, variate_id, prediction_mask (+ others).
			Returns token-level representations [N_tokens, d] or [B, L, d]. We normalize to [N_tokens, d].
		"""
		self._last_repr = None
		out = self.backbone(
			batch_dict["target"],           # tokenized / patchified by Collate
			batch_dict["observed_mask"],
			batch_dict["sample_id"],
			batch_dict["time_id"],
			batch_dict["variate_id"],
			batch_dict["prediction_mask"],
			True,                           # training_mode
		)

		# Prefer the hook
		if isinstance(self._last_repr, torch.Tensor):
			reprs = self._last_repr
		elif isinstance(out, dict):
			reprs = out.get("reprs") or out.get("hidden_states") or out.get("x")
		elif isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
			reprs = out[0]
		else:
			raise RuntimeError("Could not retrieve representations from Moirai backbone output.")

		# Normalize to [N_tokens, d]
		if reprs.dim() == 3:
			B, L, D = reprs.shape
			reprs = reprs.reshape(B*L, D)
		
		return reprs
		
	def _pool_by_sample(self, reprs: torch.Tensor, sample_id: torch.Tensor) -> torch.Tensor:
		"""
			Group tokens by sample_id → mean pool → [B, d].
			sample_id: [N_tokens] or [B,L] identifying which original sample each token belongs to.
		"""
		# Flatten sample_id to [N_tokens]
		if sample_id.dim() > 1:
			sample_id = sample_id.reshape(-1)
        
		B = int(sample_id.max().item()) + 1
		d = reprs.size(-1)
		pooled = torch.zeros(B, d, device=reprs.device, dtype=reprs.dtype)
		counts = torch.zeros(B, device=reprs.device, dtype=torch.long)
		# Simple scatter-mean without extra deps
		for sid in range(B):
			mask = (sample_id == sid)
			if mask.any():
				pooled[sid] = reprs[mask].mean(dim=0)
				counts[sid] = mask.sum()

		return pooled  # [B,d]

	def _backbone_forward_old(self, x: torch.Tensor) -> torch.Tensor:
		self._last_repr = None
		
		# Try raw [B,T,C]
		try:
			out = self.backbone(x)
		except Exception:
			# Try channel-first [B,C,T]
			try:
				out = self.backbone(x.permute(0,2,1))
			except Exception:
				# Try Uni2TS dict kwarg
				out = self.backbone(past_target=x.transpose(1,2))

		if isinstance(self._last_repr, torch.Tensor):
			return self._last_repr
		if isinstance(out, dict):
			for k in ("reprs","hidden_states","x"):
				if k in out and isinstance(out[k], torch.Tensor):
					return out[k]
		if isinstance(out, (list,tuple)) and len(out)>0 and isinstance(out[0], torch.Tensor):
			return out[0]

		raise RuntimeError("Could not obtain representations from Moirai backbone.")

	@property
	def device(self):
		try:
			return next(self.parameters()).device
		except StopIteration:
			# In case no parameters exist yet
			return torch.device("cpu")

	def forward_old(
		self,
		input: Optional[torch.Tensor] = None,
		pixel_values: Optional[torch.Tensor] = None,
		**kwargs
	) -> SequenceClassifierOutput:
		
		# Accept either key
		x = input if input is not None else pixel_values
		if x is None:
			raise ValueError("Expected `input` (time-series) or `pixel_values`.")

		reprs = self._backbone_forward_old(x)   # [B,L,d]
		reprs = reprs.transpose(1, 2)       # [B,d,L]
		pooled = self.pool(reprs).squeeze(-1)        # [B,d]
		logits = self.head(pooled)                   # [B,K]
		return SequenceClassifierOutput(logits=logits)
		
	#def forward(self, **batch) -> SequenceClassifierOutput:
	#	"""
	#		Expects the **packed** batch produced by Uni2TS collator:
	#		target, observed_mask, sample_id, time_id, variate_id, prediction_mask, labels
	#	"""
	#	reprs = self._backbone_forward(batch)
	#	pooled = self._pool_by_sample(reprs, batch["sample_id"])
	#	logits = self.head(pooled)  # [B, num_labels]
	#	return SequenceClassifierOutput(logits=logits)	
		
	def forward(self, **batch) -> SequenceClassifierOutput:
		# Call Moirai with the packed fields your version requires:
		
		print("batch")
		print(batch)
		
		out = self.backbone(
			batch["target"],          # patchified/packed by Collate
			batch["observed_mask"],
			batch["sample_id"],
			batch["time_id"],
			batch["variate_id"],
			batch["prediction_mask"],
			True,                     # training_mode
		)
		
		# out may be a tuple/dict; prefer a tensor called 'reprs' / first tensor
		if isinstance(out, dict):
			reprs = out.get("reprs") or out.get("hidden_states") or out.get("x")
		elif isinstance(out, (list, tuple)):
			reprs = out[0]
		else:
			reprs = out
		assert isinstance(reprs, torch.Tensor)

		# Flatten token grid to [N_tokens, d]
		if reprs.dim() == 3:
			B, L, D = reprs.shape
			reprs = reprs.reshape(B * L, D)

		# Pool by original sample via sample_id
		sample_id = batch["sample_id"].reshape(-1)  # [N_tokens]
		B = int(sample_id.max().item()) + 1
		d = reprs.size(-1)
		pooled = torch.zeros(B, d, device=reprs.device, dtype=reprs.dtype)
		for sid in range(B):
			mask = (sample_id == sid)
			pooled[sid] = reprs[mask].mean(dim=0)

		logits = self.head(pooled)  # [B, num_labels]
		return SequenceClassifierOutput(logits=logits)	
		
