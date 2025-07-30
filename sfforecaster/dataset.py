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

######################################
###      CLASS LABEL SCHEMA
######################################
def get_target_maps():
	""" Return dictionary of id vs targets """

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
class BaseVisDataset(Dataset):
	""" Dataset to load solar image datasets """
	
	def __init__(self, 
		filename, 
		transform=None,
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		verbose=False
	):
		self.filename= filename
		self.datalist= read_datalist(filename)
		self.transform = transform
		self.load_as_gray= load_as_gray
		self.apply_zscale= apply_zscale
		self.zscale_contrast= zscale_contrast
		self.resize= resize
		self.resize_size= resize_size
		self.verbose= verbose
		
	def load_image(self, idx):
		""" Load image as PyTorch tensor with transforms applied """
		
		# - Get image path
		item= self.datalist[idx]
		image_path= ""
		if "filepath" in item:
			image_path= item["filepaths"]
		elif "filepaths" in item:
			image_path= item["filepaths"][0]
		else:
			logger.error("No filepath/filepaths field present in input data!")
			return None	
		
		# - Load image as PyTorch tensor
		img= load_tensor(image_path)
		
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
		
		# - Load and concat image frames
		frames = [self.load_tensor(p) for p in image_paths]
		video= torch.stack(frames)  # Shape: [T, C, H, W]
		
		# - Apply transforms
		if self.transform:
			video= self.transform(video)
		
		return video
		
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
		
	def load_target(self, idx, id2target):
		""" Load single-class/single-out target """
			
		# - Get class ids
		id= self.datalist[idx]['id']
		class_id= self.id2target[id]
		
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
		
	def __len__(self):
		return len(self.datalist)
			
	def get_sample_size(self):
		return len(self.datalist)
		
		
################################################
###   VIDEO-BASED FORECASTER
################################################		
class VideoDataset(BaseVisDataset):
	""" Dataset to load solar HMI videos for single-step forecasting """
	
	def __init__(self, 
		filename, 
		transform=None, 
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		nclasses=None,
		id2target=None,
		multiout=False,
		multilabel=False,
		verbose=False
	):
		super().__init__(
			filename, 
			transform,
			load_as_gray,
			apply_zscale, zscale_contrast,
			resize, resize_size,
			verbose
		)
		
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))
		self.multiout= multiout
		self.multilabel= multilabel
		
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		video_tensor= self.load_video(idx)
		
		# - Get class ids
		if self.multiout:
			if self.multilabel:
				class_id= self.load_hotenc_targets(idx, self.id2target, self.mlb)
			else:
				class_id= self.load_targets(idx, self.id2target)
		else:
			if self.multilabel:
				class_id= self.load_hotenc_target(idx, self.id2target, self.mlb)
			else:
				class_id= self.load_target(idx, self.id2target)
		
		return video_tensor, class_id
		
################################################
###   IMAGE-BASED FORECASTER
################################################	
class ImgDataset(BaseVisDataset):
	""" Dataset to load solar HMI images for single-step forecasting """
	
	def __init__(self, 
		filename, 
		transform=None, 
		load_as_gray=False,
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		nclasses=None,
		id2target=None,
		multiout=False,
		multilabel=False,
		verbose=False
	):
		super().__init__(
			filename, 
			transform,
			load_as_gray,
			apply_zscale, zscale_contrast,
			resize, resize_size,
			verbose
		)
		
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))
		self.multiout= multiout
		self.multilabel= multilabel
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		img_tensor= self.load_image(idx)
		
		# - Get class ids
		if self.multiout:
			if self.multilabel:
				class_id= self.load_hotenc_targets(idx, self.id2target, self.mlb)
			else:
				class_id= self.load_targets(idx, self.id2target)
		else:
			if self.multilabel:
				class_id= self.load_hotenc_target(idx, self.id2target, self.mlb)
			else:
				class_id= self.load_target(idx, self.id2target)
		
		return img_tensor, class_id
		
		
################################################
###   IMAGE STACK-BASED FORECASTER
################################################		
class ImgStackDataset(BaseVisDataset):
	""" Dataset to load solar multi-channel images for single-step forecasting """
	
	def __init__(self, 
		filename, 
		transform=None, 
		apply_zscale=False,
		zscale_contrast=0.25,
		resize=False,
		resize_size=224,
		nclasses=None,
		id2target=None,
		multiout=False,
		multilabel=False,
		verbose=False
	):
		super().__init__(
			filename=filename, 
			transform=transform,
			load_as_gray=True, # load each image as [1,H,W]
			apply_zscale=apply_zscale, zscale_contrast=zscale_contrast,
			resize=resize, resize_size=resize_size,
			verbose=verbose
		)
		self.nclasses= nclasses
		self.id2target= id2target
		self.mlb = MultiLabelBinarizer(classes=np.arange(0, self.nclasses))
		self.multiout= multiout
		self.multilabel= multilabel
		
	def __getitem__(self, idx):
		""" Iterator providing training data (pixel_values + labels) """
		
		# - Load image at index as tensor 
		imgstack_tensor= self.load_image_stack(idx)
		
		# - Get class ids
		if self.multiout:
			if self.multilabel:
				class_id= self.load_hotenc_targets(idx, self.id2target, self.mlb)
			else:
				class_id= self.load_targets(idx, self.id2target)
		else:
			if self.multilabel:
				class_id= self.load_hotenc_target(idx, self.id2target, self.mlb)
			else:
				class_id= self.load_target(idx, self.id2target)
		
		return imgstack_tensor, class_id
		
