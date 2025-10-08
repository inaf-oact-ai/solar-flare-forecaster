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
from contextlib import nullcontext
import types

# - TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification
from transformers import ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoImageProcessor, AutoModelForImageClassification

# - MODULE
from sfforecaster.utils import *
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
class CoralOrdinalHead(torch.nn.Module):
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
		super().__init__(num_labels=num_labels, **kwargs)
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
		freeze_backbone: bool = False,
		patching_mode: str = "time_only" # "time_only" | "time_variate"
	):
		super().__init__()
		self.backbone = _M.from_pretrained(pretrained_name)
		
		assert patching_mode in ("time_only", "time_variate")
		self.patching_mode = patching_mode
		
		d_model = getattr(self.backbone, "d_model", 384)
		self.config = MoiraiTSConfig(num_labels=num_labels, d_model=d_model)
		#print("self.config.num_labels")
		#print(self.config.num_labels)

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
		
		#print("batch")
		#print(batch)
		
		x  = batch["target"]          # [B, L, C]
		obs= batch["observed_mask"]   # [B, L, C] (or [B, L] in some builds)
		
		# 1) read patch_size & expected input width of ts_embed
		patch_size = getattr(self.backbone, "patch_size", 16)  
		in_proj = getattr(self.backbone, "in_proj", None)
		if in_proj is None or not hasattr(in_proj, "hidden_layer"):
			raise RuntimeError("Backbone has no in_proj.hidden_layer; cannot infer expected input width.")
		# PyTorch Linear weight is [out_features, in_features]
		Wexp = in_proj.hidden_layer.in_features    # ← this is 384 in your env
    
		#print("patch_size")
		#print(patch_size)
		#print("in_proj")
		#print(in_proj)
		#print("Wexp")
		#print(Wexp)
		
		# 2) ts_embed concatenates (values + mask) → factor = 2
		concat_factor = 2
		Creq = Wexp // (concat_factor * patch_size)  # e.g. 32 / (2*16) = 1

		B, L, C = x.shape
		device, dtype = x.device, x.dtype
		
		# 3) make obs 3-D if needed
		if obs.dim() == 2:
			obs = obs.unsqueeze(-1).expand(B, L, C)

		# 4) pad/truncate channels to Creq, depending on patching mode
		#if C < Creq:
		#	pad = torch.zeros(B, L, Creq - C, dtype=dtype, device=device)
		#	x   = torch.cat([x, pad], dim=-1)
		#	obs = torch.cat([obs, torch.zeros_like(pad, dtype=torch.bool)], dim=-1)
		#elif C > Creq:
		#	x   = x[..., :Creq]
		#	obs = obs[..., :Creq]
			
		# 4) Channel handling depends on patching mode
		if self.patching_mode == "time_only":
			# Mix channels inside each temporal patch.
			if C < Creq:
				pad = torch.zeros(B, L, Creq - C, dtype=dtype, device=device)
				x   = torch.cat([x, pad], dim=-1)
				obs = torch.cat([obs, torch.zeros_like(pad, dtype=torch.bool)], dim=-1)
			elif C > Creq:
				logger.warning(f"Truncating C={C} variates to {Creq}, last {C-Creq} variates will be ignored, need to modify patch_size to include all variates with time_only patching (better switch to time_variate) ...")
				x   = x[..., :Creq]
				obs = obs[..., :Creq]			
		else:
			# "time_variate": preserve ALL variates as separate tokens (expect Creq≈1 for Moirai2-small).
			# We do NOT truncate variates here; we will expand token length by C below.
			if Creq != 1:
				# Defensive check: Moirai2-small (Wexp=32, ps=16) implies Creq=1. Warn but proceed.
				if not hasattr(self, "_warned_creq"):
					logger.warning(f"Time_variate expects Creq=1; got Creq={Creq}. Proceeding, but per-variate tokens will still use last-dim size (patch_size*Creq).")
					self._warned_creq = True		

		#print("obs")
		#print(obs.shape)
		
		# 5) truncate L to multiple of patch_size and patchify
		Lp = (L // patch_size) * patch_size
		if Lp == 0:
			raise RuntimeError(f"Sequence too short for patch_size={patch_size}.")
		if Lp != L:
			x   = x[:, :Lp, :]
			obs = obs[:, :Lp, :]

		Ltok = Lp // patch_size
		
		#x    = x.view(B, Ltok, patch_size * Creq).contiguous()       # [B, L', P*Creq]
		#obs  = obs.view(B, Ltok, patch_size * Creq).contiguous()     # [B, L', P*Creq] (bool)
		
		if self.patching_mode == "time_only":
			# [B, L, Creq] -> [B, L', ps*Creq]
			x_view   = x[:, :Lp, :].contiguous().view(B, Ltok, patch_size * Creq)
			obs_view = obs[:, :Lp, :].contiguous().view(B, Ltok, patch_size * Creq)
			x_tok, obs_tok = x_view, obs_view                       # [B, L', P]
			N = Ltok
			# IDs
			time_id   = torch.arange(Ltok, device=device).view(1, -1).expand(B, Ltok)
			sample_id = torch.arange(B,    device=device).view(-1, 1).expand(B, Ltok)
			variate_id= torch.zeros(B, Ltok, dtype=torch.long, device=device)
		else:
			# Per-variate tokens:
			# reshape to [B, L', ps, C] -> permute to [B, C, L', ps] -> flatten to [B, C*L', ps*Creq]
			x_blk   = x[:, :Lp, :].contiguous().view(B, Ltok, patch_size, C).permute(0, 3, 1, 2).contiguous()
			obs_blk = obs[:, :Lp, :].contiguous().view(B, Ltok, patch_size, C).permute(0, 3, 1, 2).contiguous()
			x_tok   = x_blk.view(B, C * Ltok, patch_size * Creq)     # typically last dim == patch_size
			obs_tok = obs_blk.view(B, C * Ltok, patch_size * Creq)
			N = C * Ltok
			# IDs
			# time ids repeat for each variate; variate ids repeat Ltok times
			time_row  = torch.arange(Ltok, device=device).repeat(C)                 # [C*L']
			var_row   = torch.arange(C, device=device).repeat_interleave(Ltok)      # [C*L']
			time_id   = time_row.view(1, -1).expand(B, N).contiguous()
			variate_id= var_row.view(1, -1).expand(B, N).contiguous().long()
			sample_id = torch.arange(B, device=device).view(-1, 1).expand(B, N)

		# 6) rebuild ids/masks to match L'
		#time_id         = torch.arange(Ltok, device=device).view(1, -1).expand(B, Ltok)
		#sample_id       = torch.arange(B,    device=device).view(-1, 1).expand(B, Ltok)
		#variate_id      = torch.zeros(B, Ltok, dtype=torch.long, device=device)
		#prediction_mask = torch.zeros(B, Ltok, dtype=torch.bool, device=device)
		
		# 6) prediction mask shaped to token length
		prediction_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

		#batch["target"]          = x
		#batch["observed_mask"]   = obs
		batch["target"]          = x_tok
		batch["observed_mask"]   = obs_tok
		batch["time_id"]         = time_id
		batch["sample_id"]       = sample_id
		batch["variate_id"]      = variate_id
		batch["prediction_mask"] = prediction_mask

		# Optional one-time prints
		if not hasattr(self, "_dbg_geom"):
			print("patch_size", patch_size)
			print("Wexp (hidden_layer.in_features)", Wexp)
			print("Creq", Creq)
			print("→ target:", tuple(x_tok.shape))
			print("→ obs   :", tuple(obs_tok.shape))
			print("→ ids   :", tuple(sample_id.shape), tuple(time_id.shape))
			self._dbg_geom = True
		
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
		
		#print("type(out)")
		#print(type(out))
		
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

		#print("reprs.dim()")
		#print(reprs.dim())
		#print(reprs.shape)

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
			#print("self.num_labels")
			#print(self.num_labels)
			self.classifier = torch.nn.Linear(in_dim, self.num_labels).to(pooled.device)

		if not hasattr(self, "_dbg_done"):
			print("[MoiraiTS] reprs:", tuple(reprs.shape))
			print("[MoiraiTS] pooled:", tuple(pooled.shape))
			print("[MoiraiTS] head in/out:", self.classifier.in_features, "->", self.classifier.out_features)
			self._dbg_done = True

		logits = self.classifier(pooled)  # [B, num_labels]
		return SequenceClassifierOutput(logits=logits)
		
###############################################
###   IMAGE-TS HYBRID MODEL
###############################################
class GlobalFeatHead(torch.nn.Module):
	"""
		Takes whatever the backbone returns and produces a single vector [B, D] per image.
		Handles common HF backbones (ViT, ConvNeXt, SigLIP vision towers, etc.).
	"""
	def __init__(self, expected_hidden_size: int | None = None):
		super().__init__()
		self.expected_hidden_size = expected_hidden_size
		# Fallback global pooling for 4D features [B, C, H', W']
		self.gap = nn.AdaptiveAvgPool2d((1, 1))

	def forward(self, backbone_outputs):
		"""
			Accepts a ModelOutput, tuple, or Tensor.
			Returns a 2D tensor [B, D].
		"""
		out = backbone_outputs

		# Case 1: HF ModelOutput with attributes (e.g., ViT/ConvNeXt)
		if hasattr(out, "pooler_output") and out.pooler_output is not None:
			# [B, D]
			return out.pooler_output

		if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
			x = out.last_hidden_state  # [B, N, D]
			# If there's a CLS token, it's typically at index 0; mean-pool is more stable across models.
			# Mean over tokens (excluding CLS if you prefer: x[:, 1:, :].mean(dim=1))
			return x.mean(dim=1)  # [B, D]

		# Case 2: the backbone returned a raw Tensor
		if isinstance(out, torch.Tensor):
			if out.dim() == 4:
				# [B, C, H, W] -> GAP -> [B, C]
				x = self.gap(out).squeeze(-1).squeeze(-2)
				return x
			elif out.dim() == 3:
				# [B, N, D] -> mean over tokens
				return out.mean(dim=1)
			elif out.dim() == 2:
				# already [B, D]
				return out
			else:
				raise ValueError(f"Unsupported tensor shape from backbone: {tuple(out.shape)}")

		# Case 3: tuple/list
		if isinstance(out, (tuple, list)) and len(out) > 0:
			maybe_tensor = out[0]
			if isinstance(maybe_tensor, torch.Tensor):
				return self.forward(maybe_tensor)
			# or recurse on a simple object that has attributes
			maybe_obj = out[0]
			if hasattr(maybe_obj, "last_hidden_state") or hasattr(maybe_obj, "pooler_output"):
				return self.forward(maybe_obj)

		raise ValueError("Unrecognized backbone output type; cannot extract a global feature vector.")


class ImageEncoderWrapper(torch.nn.Module):
	"""
		Wraps a HF AutoModelForImageClassification to expose a pure-vision encoder that
		returns per-image embeddings via GlobalFeatHead.
	"""
	def __init__(
		self, 
		model_name: str, 
		freeze_backbone: bool = False, 
		max_freeze_layer_id: int = -1, 
		trust_remote_code: bool = True
	):
		super().__init__()
		
		# - Set options
		self.freeze_backbone= freeze_backbone
		self.max_freeze_layer_id= max_freeze_layer_id
		
		# - Load entire model
		#self.full_model = AutoModelForImageClassification.from_pretrained(
		#	model_name, trust_remote_code=trust_remote_code
		#)
		_fm = AutoModelForImageClassification.from_pretrained(
			model_name, trust_remote_code=trust_remote_code
		)
		
		# - Load image processor
		self.image_processor = AutoImageProcessor.from_pretrained(model_name)
		
		# - Retrieve encoder
		#self.encoder = self._extract_encoder(self.full_model)
		self.encoder = self._extract_encoder(_fm)

		# - Create head
		self.feat_head = GlobalFeatHead()

		# IMPORTANT: do NOT keep a registered reference to the full model,
		# or you’ll have duplicate parameters under two names.
		# If you want to remember which checkpoint we used, store only metadata:
		self._orig_image_model_name = model_name
		# drop the local variable so it doesn’t get attached anywhere
		del _fm


	def _freeze_encoder(self):
		""" Freeze encoder """

		encoder_name= "encoder"
		layer_search_pattern= "layers"

		for name, param in self.encoder.named_parameters():
			if name.startswith(encoder_name):
				layer_index= extract_layer_id(name, layer_search_pattern)
				if self.max_freeze_layer_id==-1 or (self.max_freeze_layer_id>=0 and layer_index!=-1 and layer_index<self.max_freeze_layer_id):
					print(f"Freezing layer {name} ...")
					param.requires_grad = False

	@staticmethod
	def _extract_encoder(model: torch.nn.Module) -> torch.nn.Module:
		"""
			Attempts to pull the vision backbone from common fields.
			Falls back to model.base_model if available, or the whole model as last resort.
		"""
		# Prefer an explicitly named vision tower if present
		for attr in ["vision_model", "base_model", "vit", "convnext", "backbone"]:
			enc = getattr(model, attr, None)
			if isinstance(enc, torch.nn.Module):
				return enc
        
		# Some implementations put the encoder as model.<classifier>.backbone etc.
		# Last resort: if the model exposes 'model' inside (common in timm-wrapped heads)
		inner = getattr(model, "model", None)
		if isinstance(inner, torch.nn.Module):
			return inner
		
		# As a final fallback, use the full model (it should still accept pixel_values)
		return model

	def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
		"""
			pixel_values: [B, C, H, W]
			returns: per-image feature vectors [B, D]
		"""
		# Many HF encoders expect 'pixel_values' kwarg
		outputs = self.encoder(pixel_values=pixel_values)
		feats = self.feat_head(outputs)  # [B, D]
		return feats
		
class ImageFeatTSClassifier(torch.nn.Module):
	"""
		Frames [B,T,C,H,W] -> per-frame embed [B,T,D] -> proj K -> [B,T,K]
		-> build Moirai batch (target/observed_mask/ids) -> Moirai backbone -> logits [B,num_labels].
	"""
	def __init__(
		self,
		image_model_name: str,
		moirai_pretrained_name: str = "Salesforce/moirai-2.0-R-small",
		num_labels: int = 4,
		proj_dim: int = 128,
		patching_mode: str = "time_variate",  # "time_only" | "time_variate"
		freeze_backbone: bool = False,
		max_freeze_layer_id: int = -1,
		freeze_img_backbone: bool = False,
		max_img_freeze_layer_id: int = -1,
		trust_remote_code: bool = True,
		layernorm_eps: float = 1e-6,
		head_dropout: float = 0.0
	):
		super().__init__()
		assert patching_mode in ("time_only", "time_variate")
		if _M is None:
			raise RuntimeError("uni2ts not available; cannot build Moirai backbone.")

		# - Set options
		self.patching_mode = patching_mode
		self.classifier = None
		self.num_labels = num_labels
		self.freeze_img_backbone= freeze_img_backbone
		self.max_img_freeze_layer_id= max_img_freeze_layer_id
		self.freeze_backbone = freeze_backbone
		self.max_freeze_layer_id= max_freeze_layer_id
		self.dropout = torch.nn.Dropout(p=head_dropout)
		
		# - Create image encoder
		self.image_enc= ImageEncoderWrapper(
			image_model_name,
			freeze_backbone=freeze_img_backbone,
			max_freeze_layer_id=max_img_freeze_layer_id,
			trust_remote_code=trust_remote_code
		)
		self.image_processor= self.image_enc.image_processor
		
		# per-timestep projection -> K
		self._proj = None          # lazy nn.Linear(D, proj_dim) once we see D
		self._proj_dim = proj_dim
		#try:
		#	self._proj = torch.nn.LazyLinear(proj_dim)        # created at init ⇒ moved by Trainer.to(device)
		#except Exception:
		#	logger.warning("Cannot create LazyLinear layer (old pytorch version?), trying to create it later ...")
		#	self._proj = None                           # fallback for older torch; see Fix B
		
		self.ln = torch.nn.LayerNorm(proj_dim, eps=layernorm_eps)

		# - Create moirai backbone + logits head (lazy)
		self.backbone = _M.from_pretrained(moirai_pretrained_name)
		
		# - Override Moirai scaler
		scaler = getattr(self.backbone, "scaler", None)
	
		def _safe_scaler_forward(self, *args, **kwargs):
			"""
				Robust packed scaler that:
					- accepts any arg signature (v1/v2),
					- computes loc/scale OUT-OF-PLACE,
					- returns fresh fp32 tensors, so later in-place ops can't break autograd.
				Inputs (varies by Uni2TS version):
					target:        [B, N, P]
					observed_mask: [B, N, P] (bool)
					... (we ignore sample_id/time_id/variate_id/prediction_mask if provided)
				Returns:
					loc, scale:    [B, N, 1] each (fp32), cloned.
			"""

			# 1) Find target & observed_mask, positionally or by name
			target = None
			observed_mask = None

			# positional
			if len(args) >= 1:
				target = args[0]
			if len(args) >= 2:
				observed_mask = args[1]

			# keywords (override if provided)
			target = kwargs.get("target", target)
			observed_mask = kwargs.get("observed_mask", observed_mask)

			if target is None:
				raise ValueError("safe scaler: 'target' not provided")
			if observed_mask is None:
				# if mask missing in this Uni2TS build, assume fully observed
				observed_mask = torch.ones_like(target, dtype=torch.bool)

			x = target
			m = observed_mask.to(dtype=x.dtype)

			# 2) Masked mean/var per token
			denom = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
			loc   = (x * m).sum(dim=-1, keepdim=True) / denom
			xc    = (x - loc) * m
			var   = (xc * xc).sum(dim=-1, keepdim=True) / denom

			# 3) Small floor; accept either attribute name across versions
			min_scale = getattr(self, "minimum_scale", None)
			if min_scale is None:
				min_scale = getattr(self, "min_scale", 1e-5)

			# 4) Out-of-place sqrt; return fresh fp32 tensors
			scale = torch.sqrt(var + float(min_scale))

			return loc.to(torch.float32).clone(), scale.to(torch.float32).clone()

		if scaler is not None and hasattr(scaler, "forward"):
			scaler.forward = types.MethodType(_safe_scaler_forward, scaler)
			try:
				sig = str(inspect.signature(scaler.forward))
			except Exception:
				sig = "<unknown>"
			logger.info(f"Scaler patch: instance forward() overridden. New signature: {sig}")
		else:
			logger.warning("Scaler patch: no 'scaler' attribute or no forward(); could not patch.")
		
		# - Freeze encoders?
		self._freeze_encoders()

		# (optional) coax repr outputs
		for attr in ("return_repr", "output_hidden_states", "return_hidden"):
			if hasattr(self.backbone, attr):
				try: setattr(self.backbone, attr, True)
				except Exception: pass

		self._last_repr = None
		self._hooked = self._register_hook_on_any(["transformer","encoder","backbone","model","net"])

		# HF-ish config for downstream code that looks at num_labels
		from transformers.configuration_utils import PretrainedConfig
		class _Cfg(PretrainedConfig): pass
		self.config = _Cfg(num_labels=num_labels)


	def _freeze_encoders(self):
		""" Freeze model components (img backbone, moirai backbone) """ 

		# - Freeze image encoder layers?
		if self.freeze_img_backbone:
			logger.info("Freezing image encoder ...")
			self.image_enc._freeze_encoder()
						
		# - Freeze Moirai encoder layers
		if self.freeze_backbone:
			logger.info("Freezing Moirai encoder ...")
			#for p in self.backbone.parameters():
			#	p.requires_grad = False

			encoder_name= "encoder"
			layer_search_pattern= "layers"

			for name, param in self.backbone.named_parameters():
				if name.startswith(encoder_name):
					layer_index= extract_layer_id(name, layer_search_pattern)
					if self.max_freeze_layer_id==-1 or (self.max_freeze_layer_id>=0 and layer_index!=-1 and layer_index<self.max_freeze_layer_id):
						print(f"Freezing Moirai layer {name} ...")
						param.requires_grad = False

	@property
	def device(self):
		try:
			return next(self.parameters()).device
		except StopIteration:
			# model with no params yet
			logger.warning("Model without params yet, setting torch.device to CPU ...")
			return torch.device("cpu")

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
		if isinstance(self._last_repr, torch.Tensor):
			return self._last_repr
		if isinstance(out, dict):
			for k in ("reprs","hidden_states","x","last_hidden_state"):
				if k in out and isinstance(out[k], torch.Tensor):
					return out[k]
		if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
			return out[0]
		if isinstance(out, torch.Tensor):
			return out
		raise RuntimeError("Could not retrieve Moirai representations.")

	#def _maybe_init_projection(self, feat_dim: int):
	#	if self._proj is None:
	#		self._proj = torch.nn.Linear(feat_dim, self._proj_dim)

	def _maybe_init_projection(self, feat_dim: int, device: torch.device, dtype: torch.dtype):
		if self._proj is None:
			self._proj = torch.nn.Linear(feat_dim, self._proj_dim)
		# ensure it’s on the same device/dtype as inputs
		self._proj.to(device=device, dtype=dtype)
	
	def _frames_to_feats(self, frames: torch.Tensor) -> torch.Tensor:
		"""frames: [B,T,C,H,W] -> per-frame feats: [B,T,D] (then project->K and LN)"""
		B, T, C, H, W = frames.shape
		feats_per_t = []
		#ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
		#with ctx:
		#	for t in range(T):
		#		f_t = self.image_enc(frames[:, t, ...])  # [B, D]
		#		feats_per_t.append(f_t)
				
		for t in range(T):
			f_t = self.image_enc(frames[:, t, ...])  # [B, D]
			feats_per_t.append(f_t)
			
		x = torch.stack(feats_per_t, dim=1)             # [B, T, D]
		D = x.size(-1)
		
		self._maybe_init_projection(D, device=x.device, dtype=x.dtype)
		
		# Project + LayerNorm in a consistent dtype and ensure no aliasing
		x = self._proj(x)                               # [B, T, K]
		# (Optional) keep LN math in fp32 and cast back if you sometimes feed fp16/bf16
		x = self.ln(x.float()).to(x.dtype)             # [B, T, K]
		x = x.clone()
		x = x.contiguous()

		# x: [B, T, D]
		#x = self._proj(x)    # LazyLinear will infer D the first time
		#x = self.ln(x)       # [B, T, K]
		
		return x

	def _pack_for_moirai(self, X: torch.Tensor):
		"""
			X: [B, L, C]  values; builds observed_mask=1, ids, and patchifies exactly like your Moirai wrapper.
		"""
		B, L, C = X.shape
		device, dtype = X.device, X.dtype

		# observed mask = ones
		obs = torch.ones(B, L, C, dtype=torch.bool, device=device)

		# read Moirai expected in_features = in_proj.hidden_layer.in_features
		patch_size = getattr(self.backbone, "patch_size", 16)
		in_proj = getattr(self.backbone, "in_proj", None)
		if in_proj is None or not hasattr(in_proj, "hidden_layer"):
			raise RuntimeError("Backbone has no in_proj.hidden_layer; cannot infer expected input width.")
		Wexp = in_proj.hidden_layer.in_features

		# values+mask concatenation factor used by ts_embed in uni2ts
		concat_factor = 2
		Creq = Wexp // (concat_factor * patch_size)     # usually 1 for Moirai2-small

		# channel handling per patching mode (same logic as your MoiraiForSequenceClassification)
		if self.patching_mode == "time_only":
			if C < Creq:
				pad = torch.zeros(B, L, Creq - C, dtype=dtype, device=device)
				X   = torch.cat([X, pad], dim=-1)
				obs = torch.cat([obs, torch.zeros_like(pad, dtype=torch.bool)], dim=-1)
			elif C > Creq:
				logger.warning(f"Truncating variates from C={C} to {Creq} for time_only patching.")
				X   = X[..., :Creq]
				obs = obs[..., :Creq]
            
			# patchify along time
			Lp = (L // patch_size) * patch_size
			if Lp == 0:
				raise RuntimeError(f"Sequence too short for patch_size={patch_size}.")
			if Lp != L:
				X   = X[:, :Lp, :]
				obs = obs[:, :Lp, :]
            
			Ltok = Lp // patch_size
			#x_tok   = X.view(B, Ltok, patch_size * Creq)
			#obs_tok = obs.view(B, Ltok, patch_size * Creq)
			x_tok   = X.view(B, Ltok, patch_size * Creq).clone()     # break aliasing to LN output
			obs_tok = obs.view(B, Ltok, patch_size * Creq).clone()

			N = Ltok
			time_id    = torch.arange(Ltok, device=device).view(1, -1).expand(B, Ltok)
			sample_id  = torch.arange(B,    device=device).view(-1, 1).expand(B, Ltok)
			variate_id = torch.zeros(B, Ltok, dtype=torch.long, device=device)

		else:  # "time_variate" → per-variate tokens
			if Creq != 1 and not hasattr(self, "_warned_creq"):
				logger.warning(f"time_variate expects Creq=1; got Creq={Creq}. Proceeding.")
				self._warned_creq = True
			
			Lp = (L // patch_size) * patch_size
			if Lp == 0:
				raise RuntimeError(f"Sequence too short for patch_size={patch_size}.")
				
			if Lp != L:
				X   = X[:, :Lp, :]
				obs = obs[:, :Lp, :]
            
			Ltok = Lp // patch_size
			# [B,L, C] → [B, L', ps, C] → [B, C, L', ps] → [B, C*L', ps*Creq]
			#x_blk   = X.view(B, Ltok, patch_size, C).permute(0, 3, 1, 2).contiguous()
			#obs_blk = obs.view(B, Ltok, patch_size, C).permute(0, 3, 1, 2).contiguous()
			#x_tok   = x_blk.view(B, C * Ltok, patch_size * Creq)
			#obs_tok = obs_blk.view(B, C * Ltok, patch_size * Creq)
			x_blk   = X.view(B, Ltok, patch_size, C).permute(0, 3, 1, 2).contiguous()
			obs_blk = obs.view(B, Ltok, patch_size, C).permute(0, 3, 1, 2).contiguous()
			x_tok   = x_blk.view(B, C * Ltok, patch_size * Creq).clone()
			obs_tok = obs_blk.view(B, C * Ltok, patch_size * Creq).clone()
			
			N = C * Ltok
			time_row   = torch.arange(Ltok, device=device).repeat(C)
			var_row    = torch.arange(C, device=device).repeat_interleave(Ltok)
			time_id    = time_row.view(1, -1).expand(B, N)
			variate_id = var_row.view(1, -1).expand(B, N).long()
			sample_id  = torch.arange(B, device=device).view(-1, 1).expand(B, N)

		prediction_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

		return {
			"target": x_tok,                # [B, N, patch_size*Creq]
			"observed_mask": obs_tok,       # [B, N, patch_size*Creq] (bool)
			"sample_id": sample_id,         # [B, N]
			"time_id": time_id,             # [B, N]
			"variate_id": variate_id,       # [B, N]
			"prediction_mask": prediction_mask,  # [B, N]
		}

	def forward(self, pixel_values: torch.Tensor, extra_ts: torch.Tensor | None = None, labels=None):
		"""
			pixel_values: [B,T,C,H,W]
			extra_ts: optional extra covariates [B, C_extra, T] to concat on channels (time aligned)
		"""

		# 1) frames -> [B,T,K]
		feat_seq = self._frames_to_feats(pixel_values)         # [B, T, K]
        
		# 2) concat extra covariates on channel axis (→ [B,T,K+C_extra])
		if extra_ts is not None:
			# Keep a single dtype across the sequence to avoid mixed fp64/fp32 graphs
			extra_ts = extra_ts.to(feat_seq.dtype)

			if extra_ts.size(-1) != feat_seq.size(1):
				raise ValueError(f"extra_ts T={extra_ts.size(-1)} != frames T={feat_seq.size(1)}")
			# extra_ts is [B, C_extra, T] → [B, T, C_extra]
			extra_seq = extra_ts.transpose(1, 2).contiguous()
			feat_seq = torch.cat([feat_seq, extra_seq], dim=-1)  # [B, T, K+C_extra]

		# 3) pack for Moirai (build target/observed_mask/ids and patchify)
		packed = self._pack_for_moirai(feat_seq)               # dict of tensors on same device

		###############
		##  DEBUG 
		###############
		#packed["target"] = packed["target"].detach().clone()
		###################

		# 4) call backbone
		out = self.backbone(
			packed["target"],
			packed["observed_mask"],
			packed["sample_id"],
			packed["time_id"],
			packed["variate_id"],
			packed["prediction_mask"],
			True,
		)

		reprs = self._get_reprs(out)                            # [B, N, D] or [B*N, D]
		if reprs.dim() == 2:
			B, N = packed["sample_id"].shape
			reprs = reprs.view(B, N, -1)

		# 5) masked mean over tokens (valid = observed & not prediction)
		obs_tok = packed["observed_mask"]
		valid_obs = obs_tok.any(dim=-1) if obs_tok.dim() == 3 else obs_tok
		valid = valid_obs & (~packed["prediction_mask"])
		weights = valid.float()
		den = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
		pooled = (reprs * weights.unsqueeze(-1)).sum(dim=1) / den   # [B, D_repr]


		# 6) lazy classifier
		if (self.classifier is None) or (self.classifier.in_features != pooled.size(-1)):
			self.classifier = torch.nn.Linear(pooled.size(-1), self.num_labels).to(pooled.device)

		#logits = self.classifier(pooled)                         # [B, num_labels]
		logits = self.classifier(self.dropout(pooled))
		
		return SequenceClassifierOutput(logits=logits)
		
