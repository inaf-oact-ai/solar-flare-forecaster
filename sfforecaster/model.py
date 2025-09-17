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

		#if freeze_backbone:
		#	for p in self.backbone.parameters():
		#		p.requires_grad = False

		# - Try to make the backbone emit representations
		#for attr in ("return_repr", "output_hidden_states", "return_hidden"):
		#	if hasattr(self.backbone, attr):
		#		try: setattr(self.backbone, attr, True)
		#		except Exception: pass

		#self._last_repr = None
		#self._hooked = self._register_hook_on_any(["transformer","encoder","backbone","model","net"])

		#self.pool = torch.nn.AdaptiveAvgPool1d(1)
		#self.head = torch.nn.Sequential(
		#	torch.nn.Linear(d_model, d_model),
		#	torch.nn.ReLU(),
		#	torch.nn.Dropout(0.1),
		#	torch.nn.Linear(d_model, num_labels),
		#)
		
		
		# - Try to make the backbone emit representations
		for attr in ("return_repr", "output_hidden_states", "return_hidden"):
			if hasattr(self.backbone, attr):
				try: setattr(self.backbone, attr, True)
				except Exception: pass

		self._last_repr = None
		self._hooked = self._register_hook_on_any(["transformer","encoder","backbone","model","net"])

		# Lazy head: we don't assume the rep dim. We'll create it on the first forward.
		self.classifier = None          # nn.Linear will be created lazily
		self.num_labels = num_labels
		

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
		
	def _get_reprs(self, out):
		# Prefer the hook-captured tensor
		if isinstance(self._last_repr, torch.Tensor):
			return self._last_repr
		# Otherwise, probe common keys/tuple
		if isinstance(out, dict):
			for k in ("reprs", "hidden_states", "x", "last_hidden_state"):
				if k in out and isinstance(out[k], torch.Tensor):
					return out[k]
		if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
			return out[0]
		if isinstance(out, torch.Tensor):
			return out
		raise RuntimeError("Could not retrieve backbone representations (reprs).")
		
	
	@property
	def device(self):
		try:
			return next(self.parameters()).device
		except StopIteration:
			# In case no parameters exist yet
			return torch.device("cpu")

		
	def forward(self, **batch) -> SequenceClassifierOutput:
		# Call Moirai with the packed fields your version requires:
		
		print("batch")
		print(batch)
		
		x  = batch["target"]          # [B, L, C]
		obs= batch["observed_mask"]   # [B, L, C] (or [B, L] in some builds)
		
		# 1) read backbone expectations
		# infer the in_features the embed wants (384 for moirai-2.0-R-small)
		patch_size = getattr(self.backbone, "patch_size", 32)
		try:
			in_features = self.backbone.in_proj.weight.shape[1]  # [out, in]
		except Exception:
			# fallback: commonly d_ff
			in_features = getattr(self.backbone, "d_ff", patch_size * x.size(-1))
		
		
		print("patch_size")
		print(patch_size)
		print("in_features")
		print(in_features)
		
		# expected variates (channels)
		Creq = max(1, in_features // patch_size)
		print("Creq")
		print(Creq)
		
		B, L, C = x.shape
		device  = x.device
		dtype   = x.dtype
		
		# 2) pad channels if you have fewer than expected
		if C < Creq:
			pad = torch.zeros(B, L, Creq - C, dtype=dtype, device=device)
			x   = torch.cat([x, pad], dim=-1)                       # [B, L, Creq]
			if obs.dim() == 3:
				# padded channels are "unobserved"
				obs_pad = torch.zeros(B, L, Creq - C, dtype=torch.bool, device=device)
				obs     = torch.cat([obs, obs_pad], dim=-1)         # [B, L, Creq]
		elif C > Creq:
			# If you ever pass more than expected, truncate (or decide a mapping)
			x   = x[..., :Creq]
			if obs.dim() == 3:
				obs = obs[..., :Creq]
		
		print("obs")
		print(obs.shape)
		
		# 3) make sure observed_mask is 3-D so we can patchify it
		if obs.dim() == 2:
			obs = obs.unsqueeze(-1).expand(B, L, Creq)  # [B, L, Creq]
		
		print("obs")
		print(obs.shape)
		
		# 4) truncate L to multiple of patch_size and patchify
		Lp = (L // patch_size) * patch_size
		if Lp == 0:
			raise RuntimeError("Sequence too short for the chosen patch_size.")
		if Lp != L:
			x   = x[:, :Lp, :]
			obs = obs[:, :Lp, :]
		
		print("Lp")
		print(Lp)
		
		# patchify: [B, Lp, Creq] → [B, L', patch_size*Creq], with L' = Lp/ps
		Ltok = Lp // patch_size
		x    = x.view(B, Ltok, patch_size * Creq).contiguous()        # float
		obs  = obs.view(B, Ltok, patch_size * Creq).contiguous()      # bool

		# 5) rebuild the id/mask tensors so shapes match [B, Ltok]
		time_id         = torch.arange(Ltok, device=device).view(1, -1).expand(B, Ltok)
		sample_id       = torch.arange(B,    device=device).view(-1, 1).expand(B, Ltok)
		variate_id      = torch.zeros(B, Ltok, dtype=torch.long, device=device)
		prediction_mask = torch.zeros(B, Ltok, dtype=torch.bool, device=device)

		# 6) store back into the batch dict
		batch["target"]           = x
		batch["observed_mask"]    = obs
		batch["time_id"]          = time_id
		batch["sample_id"]        = sample_id
		batch["variate_id"]       = variate_id
		batch["prediction_mask"]  = prediction_mask
		
		print("→ target:", tuple(batch["target"].shape))
		print("→ obs   :", tuple(batch["observed_mask"].shape))
		print("→ ids   :", tuple(batch["sample_id"].shape), tuple(batch["time_id"].shape))
		print("→ pmask :", tuple(batch["prediction_mask"].shape))
		print("→ expected in_features:", in_features, "patch_size:", patch_size, "Creq:", Creq)
		
		# ---- now call the backbone with aligned shapes ----
		out = self.backbone(
			batch["target"],          # [B, Ltok, patch_size*Creq] → last dim == in_features
			batch["observed_mask"],   # [B, Ltok, patch_size*Creq] (bool)
			batch["sample_id"],       # [B, Ltok]
			batch["time_id"],         # [B, Ltok]
			batch["variate_id"],      # [B, Ltok]
			batch["prediction_mask"], # [B, Ltok]
			True,
		)
		
		#out = self.backbone(
		#	batch["target"],          # [B, L, P]
		#	batch["observed_mask"],   # [B, L, P]
		#	batch["sample_id"],       # [B, L]
		#	batch["time_id"],         # [B, L]
		#	batch["variate_id"],      # [B, L]
		#	batch["prediction_mask"], # [B, L]
		#	True,
		#)
		
		print("type(out)")
		print(type(out))
		
		reprs = self._get_reprs(out)    # expect [B, L, D_repr] or [B*L, D_repr]
        
		# Normalize to [B, L, D_repr]
		if reprs.dim() == 2:
			# infer B,L from batch ids
			B, L = batch["sample_id"].shape
			D = reprs.size(-1)
			reprs = reprs.view(B, L, D)
		elif reprs.dim() == 3:
			B, L, D = reprs.shape
		else:
 			raise RuntimeError(f"Unexpected reprs shape: {tuple(reprs.shape)}")

		print("reprs.dim()")
		print(reprs.dim())
		print(reprs.shape)

		# Valid timestep mask: observed anywhere in patch and not in prediction window
		obs = batch["observed_mask"]
		valid_obs = obs.any(dim=-1) if obs.dim() == 3 else obs  # [B,L]
		valid = valid_obs & (~batch["prediction_mask"])          # [B,L]
		weights = valid.float()
		den = weights.sum(dim=1, keepdim=True).clamp_min(1.0)    # [B,1]

		# Masked mean over time → [B, D_repr]
		pooled = (reprs * weights.unsqueeze(-1)).sum(dim=1) / den

		# Lazily build/resize the classifier if needed
		if (self.classifier is None) or (self.classifier.in_features != pooled.size(-1)):
			in_dim = pooled.size(-1)
			# Simple, stable head; feel free to swap back to a 2-layer MLP if you prefer
			self.classifier = torch.nn.Linear(in_dim, self.num_labels).to(pooled.device)

		if not hasattr(self, "_dbg_done"):
			print("[MoiraiTS] reprs:", tuple(reprs.shape))
			print("[MoiraiTS] pooled:", tuple(pooled.shape))
			print("[MoiraiTS] head in/out:", self.classifier.in_features, "->", self.classifier.out_features)
			self._dbg_done = True

		logits = self.classifier(pooled)  # [B, num_labels]
		return SequenceClassifierOutput(logits=logits)
		
