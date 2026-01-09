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

# - PIL
import PIL
from PIL import Image

# - SKLEARN
from sklearn.preprocessing import MultiLabelBinarizer

# - TORCH
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# - SFFORECASTER
from sfforecaster.utils import *
from sfforecaster import logger

######################################
###      CLASS LABEL SCHEMA
######################################
def get_target_maps(binary=False, flare_thr="C"):
	""" Return dictionary of id vs targets """

	if binary:
		id2target= {
			0: 0, # NONE
			1: 1,  # FLARE 
		}
		if flare_thr=="C":
			id2label= {
				0: "NONE",
				1: "C+",
			}
		elif flare_thr=="M":
			id2label= {
				0: "NONE",
				1: "M+",
			}
		else:
			id2label= {
				0: "NONE",
				1: "FLARE",
			}
		
	else:
		id2target= {
			0: 0, # NONE
			1: 1,  # C FLARE
			2: 2,  # M FLARE
			3: 3,  # X FLARE
		}
			
		id2label= {
			0: "NONE",
			1: "C",
			2: "M",
			3: "X",
		}
		
	# - Compute reverse dict
	label2id= {v: k for k, v in id2label.items()}
	
	return id2label, label2id, id2target	


######################################
###      DATASET BASE CLASS
######################################
class BaseDataset(Dataset):
	""" Dataset to load solar datasets """

	def __init__(self, 
		filename, 
		transform=None,
		verbose=False
	):
		self.filename= filename
		self.datalist= read_datalist(filename)
		self.transform = transform
		self.verbose= verbose

	def load_flareid(self, idx):
		""" Load single-class/single-out target """
			
		return self.datalist[idx]['flare_id']
		
	def load_target(self, idx, id2target):
		""" Load single-class/single-out target """
			
		# - Get class ids
		id= self.datalist[idx]['id']
		class_id= id2target[id]
		
		return class_id
		
	def load_hotenc_target(self, idx, id2target, mlb):
		""" Load single-class/single-out target as hot-encoding """
			
		# - Get class ids
		class_id= self.load_target(idx, id2target)
		
		# - Get class id (hot encoding)
		class_ids_hotenc= mlb.fit_transform([[class_id]])
		class_ids_hotenc = [j for sub in class_ids_hotenc for j in sub]
		class_ids_hotenc= torch.from_numpy(np.array(class_ids_hotenc).astype(np.float32))
		
		return class_ids_hotenc
		
	#def load_ordinal_target(self, idx):
	#	""" Load ordinal target """
	#		
	#	# - Get flare id
	#	flare_id= self.datalist[idx]['flare_id']
	#
	#	# - Get target depending on flare thresholds
	#	geC = (flare_id >= 1)
	#	geM = (flare_id >= 2)
	#	geX = (flare_id >= 3)
	#	thresholds= torch.from_numpy(np.array([geC, geM, geX]).astype(np.float32))
	#    
	#	return thresholds
		
	def load_ordinal_target(self, idx, nclasses):
		""" Load ordinal target """
		
		flare_id = int(self.datalist[idx]['flare_id']) 
		K = int(nclasses)
		thresholds = [(flare_id >= k) for k in range(1, K)]

		return torch.tensor(thresholds, dtype=torch.float32)
		
	def load_targets(self, idx, id2target):
		""" Load single-class/single-out target """
			
		# - Get class ids
		ids= self.datalist[idx]['ids']
		class_ids= [id2target[id] for id in ids]
		return class_ids
		
	def load_hotenc_targets(self, idx, id2target, mlb):
		""" Load multi-class/multi-out targets """	
		
		# - Get class ids
		class_ids= self.load_targets(idx, id2target)
		
		# - Get class id (hot encoding)
		class_ids_hotenc= [mlb.fit_transform([[id]]) for id in class_ids]
		class_ids_hotenc = [j for sub in class_ids_hotenc for j in sub]
		class_ids_hotenc= torch.from_numpy(np.array(class_ids_hotenc).astype(np.float32))
		
		return class_ids_hotenc
		
	def load_image_info(self, idx):
		""" Load image metadata """
		return self.datalist[idx]
		
	def compute_class_weights(
		self, 
		num_classes, 
		id2target, 
		scheme="balanced", 
		normalize=True,
		binary=False,
		positive_label=1,
		laplace=1.0
	):
		""" Compute class weights from dataset """
    
		# - Collect labels
		ys = []
		for i in range(len(self.datalist)):
			y= self.load_target(i, id2target)
			ys.append(int(y))
		
		counts = np.bincount(ys, minlength=num_classes).astype(float)
		
		print("counts")
		print(counts)

		# --- Binary path: also provide BCE pos_weight + optional per-sample weights
		if binary and num_classes == 2:
			pos_idx = int(positive_label)
			neg_idx = 1 - pos_idx

			# Laplace smoothing to avoid divide-by-zero on rare/empty class
			pos_s = counts[pos_idx] + laplace
			neg_s = counts[neg_idx] + laplace

			# BCEWithLogitsLoss: pos_weight multiplies the positive examples in the loss
			# canonical choice ≈ N_neg / N_pos (do NOT normalize this)
			class_weights = torch.tensor([neg_s / pos_s], dtype=torch.float32) # length-1 tensor: [N_neg/N_pos]
       
		else:
			if scheme == "inverse":
				w = 1.0 / np.maximum(counts, 1.0)
			elif scheme == "inverse_v2":
				w = np.max(counts)/counts
			else:
				# "balanced" like sklearn: n_samples / (n_classes * count_c)
				n = counts.sum()
				w = n / (num_classes * np.maximum(counts, 1.0))

			# optional normalization (keeps average weight ~1)
			print("weights")
			print(w)
			
			if normalize:
				w = w * (num_classes / w.sum())
				print("weights (after norm)")
				print(w)
		
			class_weights = torch.tensor(w, dtype=torch.float32)	
		
		return class_weights
		
	def compute_ordinal_pos_weight(self, num_classes, id2target, eps=1e-12, clip_max=None, device=None):
		"""
			Build per-threshold pos_weight for ordinal/cumulative tasks directly from dataset counts.

			Classes are assumed ordered: 0 < 1 < ... < K-1 (e.g., 0=NONE, 1=C, 2=M, 3=X).
			Ordinal heads are the K-1 tasks: [y>=1], [y>=2], ..., [y>=K-1].

				pos_weight[k-1] = (# negatives for threshold k) / (# positives for threshold k)
 					= (sum_{c<k} n_c) / (sum_{c>=k} n_c)

			Args:
				- num_classes: K (e.g., 4 for NONE,C,M,X)
				- id2target: your mapping used by self.load_target
				- eps: small constant to avoid division by zero
				- clip_max: optional float to cap very large ratios (helps stability on ultra-rare classes)
				- device: optional torch device to place the tensor on

			Returns:
				- pos_weight: torch.FloatTensor of shape (K-1,), ordered for thresholds k=1..K-1
				- counts:     np.ndarray of shape (K,) with raw class counts (for logging)
		"""
    
		# - Gather counts as in compute_class_weights
		ys = []
		for i in range(len(self.datalist)):
			y = self.load_target(i, id2target)
			ys.append(int(y))
    
		counts = np.bincount(ys, minlength=num_classes).astype(float)

		K = int(num_classes)
		assert K >= 2, "Need at least 2 ordered classes for ordinal head"

		total = counts.sum()

		# cumulative sums from the right: cum_ge[k] = sum_{c>=k} n_c
		cum_ge = np.cumsum(counts[::-1])[::-1]              # shape (K,)
		n_pos = cum_ge[1:]                                   # (K-1,) positives at each threshold k=1..K-1
		n_neg = total - n_pos                                # (K-1,) negatives are the rest

		pos_weight = n_neg / np.maximum(n_pos, eps)          # (K-1,)

		if clip_max is not None:
			pos_weight = np.minimum(pos_weight, float(clip_max))

		pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)
		return pw, counts
		
		
	def compute_sample_weights(
		self, 
		num_classes, 
		id2target, 
		scheme="balanced", 
		normalize=True
	):
		"""
			Returns a list of length len(train_ds) with per-example sampling weights.
			Typically inverse frequency by class, normalized.
		"""

		# - Collect labels
		ys = []
		for i in range(len(self.datalist)):
			y= self.load_target(i, id2target)
			ys.append(int(y))
		
		counts = np.bincount(ys, minlength=num_classes).astype(float)
		n = counts.sum()

		# - Compute sample weights		
		if scheme == "inverse":
			class_w = 1.0 / np.maximum(counts, 1.0)
		elif scheme == "inverse_v2":
			class_w = np.max(counts)/counts
		else:
			# "balanced": n / (K * count_c)
			class_w = n / (num_classes * np.maximum(counts, 1.0))

		# ----- normalize class weights to mean ~ 1 across classes (optional) -----
		if normalize:
			s = class_w.sum()
			if s > 0:
				class_w = class_w * (num_classes / s)
            
		sw = [float(class_w[y]) for y in ys]
		
		return sw
		
	def compute_sample_weights_from_flareid(self, num_classes=4, scheme="balanced", normalize=True):
		"""
			Returns a list of length len(train_ds) with per-example sampling weights.
			Typically inverse frequency by class, normalized.
		"""

		# - Collect labels
		ys = []
		for i in range(len(self.datalist)):
			y= self.load_flareid(i)
			ys.append(int(y))
		
		counts = np.bincount(ys, minlength=num_classes).astype(float)
		n = counts.sum()

		if scheme == "inverse":
			class_w = 1.0 / np.maximum(counts, 1.0)
		elif scheme == "inverse_v2":
			class_w = np.max(counts)/counts
		else:
			class_w = n / (num_classes * np.maximum(counts, 1.0))

		if normalize:
			class_w = class_w * (num_classes / class_w.sum())
		sw = [class_w[y] for y in ys]
    
		return sw	
		
	def compute_mild_focal_alpha_from_dataset(
		self,
		num_classes,
		id2target,
		use_flareid= False,
		exponent= 0.5,     # use 0.5 for sqrt inverse-frequency
		cap_ratio= 10.0,   # cap at <= 10× median weight
		device= "cpu",
		dtype= torch.float32,
	):
		"""
			Returns a torch.Tensor of shape [num_classes] to be used as focal alpha.
				- Start from class frequencies (counts / total).
				- Compute inverse-frequency^exponent (e.g., 1/sqrt(freq)).
				- Normalize so mean(alpha)=1 (nice for loss scale).
				- Cap at <= cap_ratio × median(alpha).
		"""

		# - Collect labels
		ys = []
		if use_flareid:
			for i in range(len(self.datalist)):
				y= self.load_flareid(i)
				ys.append(int(y))
		else:
			for i in range(len(self.datalist)):
				y= self.load_target(i, id2target)
				ys.append(int(y))		

		counts = np.bincount(ys, minlength=num_classes).astype(float)
		total = counts.sum()
    
		# - Avoid division by zero and handle missing classes
		eps = 1e-8
		freqs = counts / max(total, 1)
		freqs = np.clip(freqs, eps, 1.0)

		# - inverse-frequency^exponent
		inv = np.power(1.0 / freqs, exponent)

		# - normalize: mean(alpha)=1 (keeps loss scale stable)
		alpha = inv / inv.mean()

		# - cap extremes: <= cap_ratio × median(alpha)
		med = np.median(alpha)
		cap = med * cap_ratio
		alpha = np.minimum(alpha, cap)

		# - (optional) tiny floor to avoid exact zeros after numeric ops
		alpha = np.maximum(alpha, 1e-6)

		# - torch tensor
		alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
		
		return alpha_t, counts
			
	def __len__(self):
		return len(self.datalist)
			
	def get_sample_size(self):
		return len(self.datalist)
		
############################################
###      DATASET BASE CLASS FOR VIS DATA
############################################
class BaseVisDataset(BaseDataset):
	""" Dataset to load solar image datasets """
	
	def __init__(
		self,
		filename, 
		transform=None,
		verbose=False,
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		apply_asinh_stretch=False,
		pmin=0.5,
		pmax=99.5,
		asinh_scale=0.5,
	):
		super().__init__(filename, transform=transform, verbose=verbose)
		self.load_as_gray= load_as_gray
		self.apply_zscale= apply_zscale
		self.zscale_contrast= zscale_contrast
		self.resize= resize
		self.resize_size= resize_size
		self.apply_asinh_stretch= apply_asinh_stretch
		self.pmin= pmin
		self.pmax= pmax
		self.asinh_scale= asinh_scale
		
	def load_image(self, idx):
		""" Load image as PyTorch tensor with transforms applied """
		
		# - Get image path
		item= self.datalist[idx]
		image_path= ""
		if "filepath" in item:
			image_path= item["filepath"]
		elif "filepaths" in item:
			image_path= item["filepaths"][0]
		else:
			logger.error("No filepath/filepaths field present in input data!")
			return None	
		
		# - Load image as PyTorch tensor
		img= self.load_tensor(image_path)
		
		# - Check for None
		if img is None:
			return None
		
		# - Apply transforms
		if self.transform:
			img = self.transform(img)
		
		return img
		
	def load_video(self, idx):
		""" Load video as PyTorch tensor with transforms applied """
		
		# - Get image paths
		item= self.datalist[idx]
		image_paths= []
		if "filepaths" in item:
			image_paths= item["filepaths"]
		else:
			logger.error("No filepaths field present in input data!")
			return None	
		
		# - Load and concat image frames as List of T tensors of shape [C,H,W]
		frames = [self.load_tensor(p) for p in image_paths]
		
		def has_bad_frames(frame_list):
			bad_frames= False	
			for t in frame_list:
				if t is None or t.numel()==0:
					bad_frames= True
					break
			return bad_frames
		
		if has_bad_frames(frames):
			logger.warning("Input frame list has one or more None/empty tensors, returning None!")
			return None
		
		# - Apply transform (should work on list [C,H,W] or tensor [T,C,H,W])
		if self.transform:
			frames= self.transform(frames)
			
			if has_bad_frames(frames):
				logger.warning("Input frame list after transform has one or more None/empty tensors, returning None!")
				return None
		
		# - Create video tensor
		#video= torch.stack(frames)  # Shape: [T, C, H, W]
		
		# - Change tensor shape as PyTorchVideo transform API requires inputs with shape: [C, T, H, W]
		#video_cthw= video.permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
		
		# - Apply transforms
		#   NB: PyTorchVideo transform API requires inputs with shape: [C, T, H, W]
		#if self.transform:
		#	video_cthw= self.transform(video_cthw)
			
		# - Convert back to list of T tensors of Shape: [C, H, W]
		#frames_transformed = list(video_cthw.permute(1, 0, 2, 3).unbind(dim=0))
		###video_transformed= torch.stack(frames_transformed)
		
		#return frames_transformed
		##return video_transformed
		return frames
		
		
	def load_image_stack(self, idx):
		""" Load multi-channel image as PyTorch tensor with transforms applied """
		
		# - Get image paths
		item= self.datalist[idx]
		image_paths= []
		if "filepaths" in item:
			image_paths= item["filepaths"]
		else:
			logger.error("No filepaths field present in input data!")
			return None	
		
		# - Load and concat image frames
		images = [self.load_tensor(p) for p in image_paths]
		image_stack= torch.cat(images, dim=0)  # Shape: [C, H, W]
		
		# - Apply transforms
		if self.transform:
			image_stack= self.transform(image_stack)
		
		return image_stack
		
	def load_npy_image(self, image_path):
		""" Load image as numpy """	
		
		# - Read image (FITS/natural image supported) and then convert to numpy either as 1D or 3-chan image, normalized to [0,1]
		if self.load_as_gray:
			img= load_img_as_npy_float(
				image_path, 
				add_chan_axis=True,
				add_batch_axis=False,
				resize=self.resize, resize_size=self.resize_size, 
				apply_zscale=self.apply_zscale, contrast=self.zscale_contrast, 
				set_nans_to_min=False,
				apply_asinh_stretch=self.apply_asinh_stretch,
				pmin=self.pmin,
				pmax=self.pmax,
				asinh_scale=self.asinh_scale,
				verbose=self.verbose
			)
		else:
			img= load_img_as_npy_rgb_float(
				image_path, 
				add_chan_axis=True,
				add_batch_axis=False,
				resize=self.resize, resize_size=self.resize_size, 
				apply_zscale=self.apply_zscale, contrast=self.zscale_contrast, 
				set_nans_to_min=False,
				apply_asinh_stretch=self.apply_asinh_stretch,
				pmin=self.pmin,
				pmax=self.pmax,
				asinh_scale=self.asinh_scale,
				verbose=self.verbose
			)
			
		# - Check for None
		if img is None:
			return None
			
		# - Replace NaN or Inf with zeros
		img[~np.isfinite(img)] = 0
		
		return img
							
	def load_tensor(self, image_path):
		""" Read image as PyTorch tensor with shape: [C, H, W]. No transforms applied. """			
		
		# - Load image as npy
		img= self.load_npy_image(image_path)
		
		# - Check for None
		if img is None:
			return None
		
		# - Convert numpy image to tensor	
		img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
		
		# - Replace NaN or Inf with zeros
		img[~torch.isfinite(img)] = 0
		
		return img
		
		
################################################
###   VIDEO-BASED FORECASTER
################################################		
class VideoDataset(BaseVisDataset):
	""" Dataset to load solar HMI videos for single-step forecasting """

	def __init__(
		self,
		filename, 
		transform=None,
		verbose=False,
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		apply_asinh_stretch=False,
		pmin=0.5,
		pmax=99.5,
		asinh_scale=0.5,
		nclasses=None,
		id2target=None,
		multiout=False,
		multilabel=False,
		ordinal=False,
	):
		super().__init__(
			filename=filename, 
			transform=transform,
			verbose=verbose,
			load_as_gray=load_as_gray,
			apply_zscale=apply_zscale, 
			zscale_contrast=zscale_contrast,
			resize=resize,
			resize_size=resize_size,
			apply_asinh_stretch=apply_asinh_stretch,
			pmin=pmin,
			pmax=pmax,
			asinh_scale=asinh_scale
		)
		
		# - Check nclasses		
		if nclasses is None:
			if id2target is not None:
				nclasses = len(id2target)
			else:
				raise ValueError("nclasses is None and id2target not provided; cannot infer number of classes.")
				
		# - Validate flags
		if multilabel and ordinal:
			raise ValueError("Invalid config: multilabel=True and ordinal=True are mutually exclusive.")
                
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))
		self.multiout= multiout
		self.multilabel= multilabel
		self.ordinal= ordinal
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		frame_tensor_list= self.load_video(idx)
		
		# - Get class ids
		if self.multiout:
			if self.multilabel:
				class_id_tensor= self.load_hotenc_targets(idx, self.id2target, self.mlb)
			elif self.ordinal:	
				raise ValueError("Ordinal training not implemented/supported for multilabel!")
			else:
				class_id= self.load_targets(idx, self.id2target)
				class_id_tensor= torch.tensor(class_id, dtype=torch.long)
		else:
			if self.multilabel:
				class_id_tensor= self.load_hotenc_target(idx, self.id2target, self.mlb)
			elif self.ordinal:
				class_id_tensor= self.load_ordinal_target(idx, self.nclasses)
			else:
				class_id= self.load_target(idx, self.id2target)
				class_id_tensor= torch.tensor(class_id, dtype=torch.long)
				
		return frame_tensor_list, class_id_tensor
		
################################################
###   IMAGE-BASED FORECASTER
################################################	
class ImgDataset(BaseVisDataset):
	""" Dataset to load solar HMI images for single-step forecasting """
	
	def __init__(self, 
		filename, 
		transform=None,
		verbose=False,
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		apply_asinh_stretch=False,
		pmin=0.5,
		pmax=99.5,
		asinh_scale=0.5,
		nclasses=None,
		id2target=None,
		multiout=False,
		multilabel=False,
		ordinal=False
	):
		super().__init__(
			filename=filename, 
			transform=transform,
			verbose=verbose,
			load_as_gray=load_as_gray,
			apply_zscale=apply_zscale, 
			zscale_contrast=zscale_contrast,
			resize=resize, 
			resize_size=resize_size,
			apply_asinh_stretch=apply_asinh_stretch,
			pmin=pmin,
			pmax=pmax,
			asinh_scale=asinh_scale
		)

		# - Check nclasses		
		if nclasses is None:
			if id2target is not None:
				nclasses = len(id2target)
			else:
				raise ValueError("nclasses is None and id2target not provided; cannot infer number of classes.")
				
		# - Validate flags
		if multilabel and ordinal:
			raise ValueError("Invalid config: multilabel=True and ordinal=True are mutually exclusive.")
            		
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))
		self.multiout= multiout
		self.multilabel= multilabel
		self.ordinal= ordinal
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		img_tensor= self.load_image(idx)
		
		# - Get class ids
		if self.multiout:
			if self.multilabel:
				class_id_tensor= self.load_hotenc_targets(idx, self.id2target, self.mlb)
			elif self.ordinal:	
				raise ValueError("Ordinal training not implemented/supported for multilabel!")
			else:
				class_id= self.load_targets(idx, self.id2target)
				class_id_tensor= torch.tensor(class_id, dtype=torch.long)
		else:
			if self.multilabel:
				class_id_tensor= self.load_hotenc_target(idx, self.id2target, self.mlb)
			elif self.ordinal:
				class_id_tensor= self.load_ordinal_target(idx, self.nclasses)
			else:
				class_id= self.load_target(idx, self.id2target)
				class_id_tensor= torch.tensor(class_id, dtype=torch.long)
		
		return img_tensor, class_id_tensor
		
		
################################################
###   IMAGE STACK-BASED FORECASTER
################################################		
class ImgStackDataset(BaseVisDataset):
	""" Dataset to load solar multi-channel images for single-step forecasting """
	
	def __init__(
		self, 
		filename, 
		transform=None,
		verbose=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		apply_asinh_stretch=False,
		pmin=0.5,
		pmax=99.5,
		asinh_scale=0.5,
		nclasses=None,
		id2target=None,
		multiout=False,
		multilabel=False,
		ordinal=False
	):
		super().__init__(
			filename=filename, 
			transform=transform,
			verbose=verbose,
			load_as_gray=True, # load each image as [1,H,W]
			apply_zscale=apply_zscale, 
			zscale_contrast=zscale_contrast,
			resize=resize, 
			resize_size=resize_size,
			apply_asinh_stretch=apply_asinh_stretch,
			pmin=pmin,
			pmax=pmax,
			asinh_scale=asinh_scale
		)
		
		# - Check nclasses		
		if nclasses is None:
			if id2target is not None:
				nclasses = len(id2target)
			else:
				raise ValueError("nclasses is None and id2target not provided; cannot infer number of classes.")
				
		# - Validate flags
		if multilabel and ordinal:
			raise ValueError("Invalid config: multilabel=True and ordinal=True are mutually exclusive.")
				
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))
		self.multiout= multiout
		self.multilabel= multilabel
		self.ordinal= ordinal
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		imgstack_tensor= self.load_image_stack(idx)
		
		# - Get class ids
		if self.multiout:
			if self.multilabel:
				class_id_tensor= self.load_hotenc_targets(idx, self.id2target, self.mlb)
			elif self.ordinal:	
				raise ValueError("Ordinal training not implemented/supported for multilabel!")
			else:
				class_id= self.load_targets(idx, self.id2target)
				class_id_tensor= torch.tensor(class_id, dtype=torch.long)
		else:
			if self.multilabel:
				class_id_tensor= self.load_hotenc_target(idx, self.id2target, self.mlb)
			elif self.ordinal:
				class_id_tensor= self.load_ordinal_target(idx, self.nclasses)	
			else:
				class_id= self.load_target(idx, self.id2target)
				class_id_tensor= torch.tensor(class_id, dtype=torch.long)
		
		return imgstack_tensor, class_id_tensor
		
		
######################################
###  TIME-SERIES DATASET BASE CLASS
######################################
class TSDataset(BaseDataset):
	""" Dataset to load solar time series datasets """
	
	def __init__(
		self, 
		filename, 
		transform=None,
		verbose=False,
		nclasses=None,
		id2target=None,
		multiout=False,
		multilabel=False,
		ordinal=False,
		data_vars=["xrs_flux_ratio", "flare_hist"],
		logstretch_vars=[False, False],
		npoints=1440
	):
		super().__init__(
			filename=filename, 
			transform=transform,
			verbose=verbose
		)
		
		# - Check nclasses		
		if nclasses is None:
			if id2target is not None:
				nclasses = len(id2target)
			else:
				raise ValueError("nclasses is None and id2target not provided; cannot infer number of classes.")
				
		# - Validate flags
		if multilabel and ordinal:
			raise ValueError("Invalid config: multilabel=True and ordinal=True are mutually exclusive.")
				
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))
		self.multiout= multiout
		self.multilabel= multilabel
		self.ordinal= ordinal
		self.data_vars= data_vars
		self.logstretch_vars= logstretch_vars
		self.data_var_stats= {}
		self.npoints= npoints
		#self.eps= 1.e-8
		
	def compute_ts_var_stats(self):
		""" Compute ts var dataset stats """	
		
		#print("len(self.data_vars)")
		#print(len(self.data_vars))
		#print("len(self.logstretch_vars)")
		#print(len(self.logstretch_vars))
		
		for k, data_var in enumerate(self.data_vars):
			data_all= []
			data_npoints= []
			for idx in range(len(self.datalist)):
				try:
					data= self.load_ts_var(idx, data_var, self.logstretch_vars[k])
				except Exception as e:
					logger.error(f"Failed to read ts data {idx} for var {data_var} (k={k})")
					raise e
				if data is None:
					logger.error(f"Failed to read ts data {idx} for var {data_var}, skipping ...")
					continue
				n= data.size
				if n!=self.npoints:
					logger.warning(f"Sample {idx} for var {data_var} has {n}!={self.npoints} points, skipping it ...")
					continue	
				data_npoints.append(data.size)
				data_all.append(data)
				
			data_all= np.concatenate(data_all)	
			data_min= np.min(data_all)
			data_max= np.max(data_all)
			data_mean= np.mean(data_all)
			data_median= np.median(data_all)
			data_std= np.std(data_all, ddof=1)
			data_npoints= np.unique(data_npoints)
			data_stats= {
				"min": float(data_min),
				"max": float(data_max),
				"mean": float(data_mean),
				"median": float(data_median),
				"data_std": float(data_std),
				"npoints": list(data_npoints)
			}
			self.data_var_stats[data_var]= data_stats
				
		
	def load_ts_var(self, idx, data_var, logstretch):
		""" Load time series for a specific variable as numpy array """
		
		# - Get ts data variable
		item= self.datalist[idx]
		data= None
		if data_var in item:
			data= np.asarray(item[data_var], dtype=np.float32).reshape(-1)
		else:
			logger.error(f"No data with name {data_var} present in input data!")
			return None
		
		# - Replace NaNs with zeros
		cond= ~np.isfinite(data)
		data[cond]= 0.0
		
		if self.verbose:
			print(f"Var {data_var} min/max: {data.min()}/{data.max()}")
		
		# - Log stretch?
		if logstretch:
			data_min= data.min()
			data= np.where(data==data_min, 0.0, np.log10(1.0 + data - data_min))
			#data= np.log10(1.0 + data - data_min)
			cond= ~np.isfinite(data)
			data[cond]= 0.0
		
			if self.verbose:
				print(f"Var {data_var} min/max (after logstretch): {data.min()}/{data.max()}")
		
		return data
	
	def load_ts_tensor(self, idx):
		""" Load time series vars as PyTorch tensors """
		
		# - Loop over data vars and build feature channels
		data_ch_list = []
		for i, data_var in enumerate(self.data_vars):
			data_ch= self.load_ts_var(idx, data_var, self.logstretch_vars[i])
			if data_ch is None:
				logger.error(f"Failed to read ts data for var {data_var}, returning None!")
				return None
			data_ch_list.append(data_ch)
			
		# - Ensure that all feature time series have the same length
		ts_lengths= [item.size for item in data_ch_list]
		ts_unique_lengths= set(ts_lengths)
		equal_length= len(ts_unique_lengths) == 1
		if not equal_length:
			logger.error(f"Feature time series have different lengths ({str(ts_unique_lengths)}), returning None!")
			return None
		
		T= ts_lengths[0]
		C= len(data_ch_list)
		
		# - Stack feature channels → [T, C]
		x_tc = np.stack(data_ch_list, axis=-1).astype(np.float32)  # [T, C]
		
		# - Convert numpy image to tensor	
		x_tensor = torch.from_numpy(x_tc).contiguous()
		
		# - Apply transforms
		if self.transform:
			x_tensor = self.transform(x_tensor)
		
		# - Replace NaN or Inf with zeros
		x_tensor[~torch.isfinite(x_tensor)] = 0
		
		return x_tensor
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load ts data at index as tensor [T, C]
		ts_tensor= self.load_ts_tensor(idx)
		
		# - Get class ids
		if self.multiout:
			if self.multilabel:
				class_id_tensor= self.load_hotenc_targets(idx, self.id2target, self.mlb)
			elif self.ordinal:	
				raise ValueError("Ordinal training not implemented/supported for multilabel!")
			else:
				class_id= self.load_targets(idx, self.id2target)
				class_id_tensor= torch.tensor(class_id, dtype=torch.long)
		else:
			if self.multilabel:
				class_id_tensor= self.load_hotenc_target(idx, self.id2target, self.mlb)
			elif self.ordinal:
				class_id_tensor= self.load_ordinal_target(idx, self.nclasses)
			else:
				class_id= self.load_target(idx, self.id2target)
				class_id_tensor= torch.tensor(class_id, dtype=torch.long)
		
		return ts_tensor, class_id_tensor	
		


######################################
###  MULTI-MODAL DATASET CLASS
######################################
class _VideoLoader(BaseVisDataset):
	"""Internal helper that reuses :class:`BaseVisDataset` methods without
	reading the JSON file again.

	The outer multimodal dataset owns `datalist`; this helper just consumes it.
	"""

	def __init__(
		self,
		datalist,
		transform=None,
		verbose=False,
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		apply_asinh_stretch=False,
		pmin=0.5,
		pmax=99.5,
		asinh_scale=0.5,
	):
		# NOTE: do NOT call BaseDataset/BaseVisDataset __init__ (it would re-read JSON)
		self.filename = None
		self.datalist = datalist
		self.transform = transform
		self.verbose = verbose
		self.load_as_gray = load_as_gray
		self.apply_zscale = apply_zscale
		self.zscale_contrast = zscale_contrast
		self.resize = resize
		self.resize_size = resize_size
		self.apply_asinh_stretch = apply_asinh_stretch
		self.pmin = pmin
		self.pmax = pmax
		self.asinh_scale = asinh_scale
		
class _TSLoader(TSDataset):
	"""Internal helper that reuses :class:`TSDataset` time-series loading without
	reading the JSON file again.

	The outer multimodal dataset owns `datalist`; this helper just consumes it.
	"""

	def __init__(
		self,
		datalist,
		transform=None,
		verbose=False,
		data_vars=("xrs_flux_ratio", "flare_hist"),
		logstretch_vars=(False, False),
		npoints=1440,
	):
		# NOTE: do NOT call BaseDataset/TSDataset __init__ (it would re-read JSON)
		self.filename = None
		self.datalist = datalist
		self.transform = transform
		self.verbose = verbose
		self.data_vars = list(data_vars)
		self.logstretch_vars = list(logstretch_vars)
		self.data_var_stats = {}
		self.npoints = npoints
		
		
class MultimodalDataset(BaseDataset):
	"""Multimodal dataset returning (video, time-series, label).

	This reuses the existing video/ts loading
	logic but keep a single source of truth for `datalist` and label handling.

	Returned sample is a dict:
		{
		  "video": <list[tensor] or tensor>,
		  "ts": <tensor [T,C]>,
		  "label": <tensor>,
		  "meta": <dict>
		}
	"""

	def __init__(
		self,
		filename,
		video_transform=None,
		ts_transform=None,
		verbose=False,
		# --- video loader options (mirrors BaseVisDataset)
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		apply_asinh_stretch=False,
		pmin=0.5,
		pmax=99.5,
		asinh_scale=0.5,
		# --- ts loader options (mirrors TSDataset)
		data_vars=("xrs_flux_ratio", "flare_hist"),
		logstretch_vars=(False, False),
		npoints=1440,
		# --- label options (mirrors VideoDataset/TSDataset)
		nclasses=None,
		id2target=None,
		multiout=False,
		multilabel=False,
		ordinal=False,
		# --- multimodal options
		require_matched=True,
	):
		super().__init__(filename=filename, transform=None, verbose=verbose)

		# - Check nclasses
		if nclasses is None:
			if id2target is not None:
				nclasses = len(id2target)
			else:
				raise ValueError("nclasses is None and id2target not provided; cannot infer number of classes.")

		# - Validate flags
		if multilabel and ordinal:
			raise ValueError("Invalid config: multilabel=True and ordinal=True are mutually exclusive.")

		self.nclasses = nclasses
		self.id2target = id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))
		self.multiout = multiout
		self.multilabel = multilabel
		self.ordinal = ordinal
		self.require_matched = require_matched

		# - Create modality loaders that reuse existing logic but share datalist
		self._video = _VideoLoader(
			datalist=self.datalist,
			transform=video_transform,
			verbose=verbose,
			load_as_gray=load_as_gray,
			apply_zscale=apply_zscale,
			zscale_contrast=zscale_contrast,
			resize=resize,
			resize_size=resize_size,
			apply_asinh_stretch=apply_asinh_stretch,
			pmin=pmin,
			pmax=pmax,
			asinh_scale=asinh_scale,
		)
		self._ts = _TSLoader(
			datalist=self.datalist,
			transform=ts_transform,
			verbose=verbose,
			data_vars=data_vars,
			logstretch_vars=logstretch_vars,
			npoints=npoints,
		)

		# - Optional filtering for matched samples
		if self.require_matched:
			self._filter_unmatched_inplace()

	def _filter_unmatched_inplace(self):
		"""Drop samples with missing video frames or ts variables."""
		filtered = []
		for s in self.datalist:
			# video: require filepaths present and non-empty
			if "filepaths" not in s or s.get("filepaths") is None or len(s.get("filepaths")) == 0:
				continue
			# ts: require each var present and not None
			ok_ts = True
			for v in self._ts.data_vars:
				if v not in s or s.get(v) is None:
					ok_ts = False
					break
			if not ok_ts:
				continue
			filtered.append(s)
		self.datalist = filtered
		# keep helpers in sync
		self._video.datalist = self.datalist
		self._ts.datalist = self.datalist

	def _load_label_tensor(self, idx):
		"""Match label logic used across existing datasets."""
		if self.multiout:
			if self.multilabel:
				return self.load_hotenc_targets(idx, self.id2target, self.mlb)
			elif self.ordinal:
				raise ValueError("Ordinal training not implemented/supported for multilabel/multiout!")
			else:
				class_id = self.load_targets(idx, self.id2target)
				return torch.tensor(class_id, dtype=torch.long)
		else:
			if self.multilabel:
				return self.load_hotenc_target(idx, self.id2target, self.mlb)
			elif self.ordinal:
				return self.load_ordinal_target(idx, self.nclasses)
			else:
				class_id = self.load_target(idx, self.id2target)
				return torch.tensor(class_id, dtype=torch.long)

	def __getitem__(self, idx):
		video = self._video.load_video(idx)
		ts = self._ts.load_ts_tensor(idx)
		if video is None or ts is None:
			# If require_matched=False you might still want to skip on the fly.
			# Returning None lets a custom collate_fn drop it.
			return None
		label = self._load_label_tensor(idx)
		meta = self.load_image_info(idx)
		return {"video": video, "ts": ts, "label": label, "meta": meta}
