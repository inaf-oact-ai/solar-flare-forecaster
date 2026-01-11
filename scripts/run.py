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
import argparse
from pathlib import Path
import json
import csv

# - SKLEARN
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

# - TORCH
import torch
##torch.autograd.set_detect_anomaly(True)  ## DEBUG
from torch.optim import AdamW
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# - TRANSFORMERS
import transformers
from transformers import Trainer
from transformers import pipeline
from transformers import AutoProcessor, AutoModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
from transformers import VideoMAEForVideoClassification, VideoMAEConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.data.data_collator import DataCollator
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers import EvalPrediction 
#import evaluate

# - SFFORECASTER
from sfforecaster.model import MultiHorizonVideoMAE, MoiraiForSequenceClassification, ImageFeatTSClassifier
from sfforecaster.utils import *
from sfforecaster.dataset import get_target_maps
from sfforecaster.dataset import VideoDataset, ImgDataset, ImgStackDataset, TSDataset
from sfforecaster.dataset import MultimodalDataset
from sfforecaster.custom_transforms import FlippingTransform, Rotate90Transform, RandomCenterCrop
from sfforecaster.custom_transforms import VideoFlipping, VideoResize, VideoNormalize, VideoRotate90, VideoRandomCenterCrop
from sfforecaster.metrics import build_multi_label_metrics, build_single_label_metrics, build_ordinal_metrics
from sfforecaster.trainer import CustomTrainer, CustomTrainerTS, TrainMetricsCallback, CudaGCCallback
from sfforecaster.trainer import VideoDataCollator, ImgDataCollator, TSDataCollator, Uni2TSBatchCollator, VideoUni2TSMultimodalCollator
from sfforecaster.model import CoralOrdinalHead
from sfforecaster.model import MultimodalConcatMLP
from sfforecaster.inference import coral_logits_to_class_probs, coral_decode_with_thresholds
from sfforecaster.inference import load_img_for_inference, load_video_for_inference, load_ts_for_inference
from sfforecaster.metrics import binary_curves_from_probs

from sfforecaster import logger

# - Configure transformer logging
transformers.utils.logging.set_verbosity(transformers.logging.DEBUG)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
    
# - Configure wandb
os.environ["WANDB_PROJECT"]= "sfforecaster"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


#### GET SCRIPT ARGS ####
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def list_of_floats(arg):
	return list(map(float, arg.split(',')))

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-datalist','--datalist', dest='datalist', required=True, type=str, help='Input data json filelist') 
	parser.add_argument('-datalist_cv','--datalist_cv', dest='datalist_cv', required=False, default="", type=str, help='Input data json filelist for validation') 
	
	# - Data loading options
	parser.add_argument('-ts_vars', '--ts_vars', dest='ts_vars', required=False, type=str, default='xrs_flux_ratio,flare_hist', action='store', help='Name of time series variables in input json data, separated by commas')
	parser.add_argument('-ts_npoints', '--ts_npoints', dest='ts_npoints', required=False, type=int, default=1440, action='store',help='Number of points in ts features (default=1440)')
	
	# - Image pre-processing options (IMAGES/VIDEOS)
	parser.add_argument('--use_model_processor', dest='use_model_processor', action='store_true', help='Use model image processor in data collator (default=false)')	
	parser.set_defaults(use_model_processor=False)
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform to input images (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('-zscale_contrast', '--zscale_contrast', dest='zscale_contrast', required=False, type=float, default=0.25, action='store', help='zscale contrast parameter (default=0.25)')
	parser.add_argument('--grayscale', dest='grayscale', action='store_true',help='Load input images in grayscale (1 chan tensor) (default=false)')	
	parser.set_defaults(grayscale=False)
	parser.add_argument('--resize', dest='resize', action='store_true', help='Resize input image before model processor. If false the model processor will resize anyway to its image size (default=false)')	
	parser.set_defaults(resize=False)
	parser.add_argument('-resize_size', '--resize_size', dest='resize_size', required=False, type=int, default=224, action='store', help='Resize size in pixels used if --resize option is enabled (default=224)')	
	parser.add_argument('--asinh_stretch', dest='asinh_stretch', action='store_true',help='Apply asinh stretch transform to input images (default=false)')	
	parser.set_defaults(asinh_stretch=False)
	parser.add_argument('-pmin', '--pmin', dest='pmin', required=False, type=float, default=0.5, action='store', help='Min percentile for asinh transform (default=0.5)')
	parser.add_argument('-pmax', '--pmax', dest='pmax', required=False, type=float, default=99.5, action='store', help='Max percentile for asinh transform (default=99.5)')
	parser.add_argument('-asinh_scale', '--asinh_scale', dest='asinh_scale', required=False, type=float, default=0.5, action='store', help='asinh_scale for asinh transform (default=0.5)')
	
	# - Image pre-processing options (TS)
	parser.add_argument('-ts_logstretchs', '--ts_logstretchs', dest='ts_logstretchs', required=False, type=str, default='0,0', action='store', help='Log stretch TS vars separated by commas (1=enable, 0=disable). Must have same dimension of ts_vars.')	
	
	# - Image augmentations options
	parser.add_argument('--add_crop_augm', dest='add_crop_augm', action='store_true', help='If enabled, add random center crop and resize (--resize_size) augmentation in training (default=false)')	
	parser.set_defaults(add_crop_augm=False)
	parser.add_argument('-min_crop_fract', '--min_crop_fract', dest='min_crop_fract', required=False, type=float, default=0.65, action='store', help='Mininum crop fraction (default=0.65).')
	
	# - Model options (GENERAL)
	parser.add_argument("--data_modality", dest='data_modality', type=str, choices=["image", "video", "ts", "multimodal"], default="image", help="Data modality model used")
	parser.add_argument('-model', '--model', dest='model', required=False, type=str, default="google/siglip-so400m-patch14-384", action='store', help='Model pretrained file name or weight path to be loaded {google/siglip-large-patch16-256, google/siglip-base-patch16-256, google/siglip-base-patch16-256-i18n, google/siglip-so400m-patch14-384, google/siglip-base-patch16-224, MCG-NJU/videomae-base, MCG-NJU/videomae-large, OpenGVLab/VideoMAEv2-Large}')
	
	parser.add_argument('--binary', dest='binary', action='store_true',help='Choose binary classification label scheme (default=false)')	
	parser.set_defaults(binary=False)
	parser.add_argument('-flare_thr', '--flare_thr', dest='flare_thr', required=False, type=str, default='M', action='store',help='Choose flare class label name: {C-->label=C+,M-->label=M+}.')
	parser.add_argument('-binary_thr', '--binary_thr', dest='binary_thr', required=False, type=float, default=0.5, action='store',help='Binary selection threshold (default=0.5).')
	
	parser.add_argument('--multilabel', dest='multilabel', action='store_true',help='Do multilabel classification (default=false)')	
	parser.set_defaults(multilabel=False)
	parser.add_argument('--multiout', dest='multiout', action='store_true',help='Do multi-step forecasting classification (default=false)')	
	parser.set_defaults(multiout=False)
	parser.add_argument('-num_horizons', '--num_horizons', dest='num_horizons', required=False, type=int, default=3, action='store',help='Number of forecasting horizons (default=3)')
	
	parser.add_argument('--ordinal', dest='ordinal', action='store_true',help='Load ordinal head model for classification (default=false)')	
	parser.set_defaults(ordinal=False)
	
	# - Model options (IMAGE/VIDEO)
	parser.add_argument('--vitloader', dest='vitloader', action='store_true', help='If enabled use ViTForImageClassification to load model otherwise AutoModelForImageClassification (default=false)')	
	parser.set_defaults(vitloader=False)
	parser.add_argument("--video_model", dest='video_model', type=str, choices=["videomae", "imgfeatts"], default="videomae", help="Video model used")
	parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true',help='Make backbone layers are non-tranable (default=false)')	
	parser.set_defaults(freeze_backbone=False)
	parser.add_argument('-max_freeze_layer_id', '--max_freeze_layer_id', dest='max_freeze_layer_id', required=False, type=int, default=-1, action='store',help='ID of the last layer kept frozen. -1 means all are frozen if --freeze_backbone option is enabled (default=-1)')
	parser.add_argument('--check_videomae_head', dest='check_videomae_head', action='store_true', help='Reinitialize videoMAE head if detected with suspicious numerica weights (default=false)')	
	parser.set_defaults(check_videomae_head=False)
	
	# - Model options (TIME SERIES)	
	parser.add_argument('-model_ts_backbone', '--model_ts_backbone', dest='model_ts_backbone', required=False, type=str, default="Salesforce/moirai-2.0-R-small", action='store', help='Model TS backbone name')
	parser.add_argument("--ts_patching_mode", dest='ts_patching_mode', type=str, choices=["time_only", "time_variate"], default="time_variate", help="Patching mode to be used with input ts variates")
	parser.add_argument('-model_ts_img_backbone', '--model_ts_img_backbone', dest='model_ts_img_backbone', required=False, type=str, default="google/siglip2-base-patch16-224", action='store', help='Model imgfeatts image backbone name')
	parser.add_argument('-proj_dim', '--proj_dim', dest='proj_dim', required=False, type=int, default=128, action='store',help='Size of linear projection layer in ImageFeatTSClassifier model (default=128)')
	
	parser.add_argument('--ts_freeze_backbone', dest='ts_freeze_backbone', action='store_true',help='Make Moirai backbone layers are non-tranable (default=false)')	
	parser.set_defaults(ts_freeze_backbone=False)
	parser.add_argument('-ts_max_freeze_layer_id', '--ts_max_freeze_layer_id', dest='ts_max_freeze_layer_id', required=False, type=int, default=-1, action='store',help='ID of the last layer kept frozen. -1 means all are frozen if --ts_freeze_backbone option is enabled (default=-1)')
	
	# - Model options (MULTIMODAL)
	parser.add_argument("--mm_fusion", dest="mm_fusion", type=str, choices=["concat_mlp"], default="concat_mlp", help="Multimodal fusion strategy")
	parser.add_argument("--mm_hidden_dim", dest="mm_hidden_dim", type=int, default=512, help="Hidden dim for multimodal fusion MLP")
	parser.add_argument("--require_matched", dest="require_matched", action="store_true", help="Use only samples with both video and TS present")
	parser.set_defaults(require_matched=False)
	parser.add_argument("--mm_ckpt", dest="mm_ckpt", required=False, type=str, default="", action="store", help="Path to multimodal checkpoint (dir/.bin/.safetensors). If set, loads fusion weights after building model.")
	parser.add_argument("--video_ckpt", dest="video_ckpt", required=False, type=str, default="", action="store", help="Path to trained video checkpoint (dir/.bin/.safetensors) to init video branch.")
	parser.add_argument("--ts_ckpt", dest="ts_ckpt", required=False, type=str, default="", action="store", help="Path to trained ts checkpoint (dir/.bin/.safetensors) to init ts branch.")
	parser.add_argument("--mm_init", dest="mm_init", required=False, type=str, default="unimodal", choices=["pretrained", "unimodal", "multimodal"], help="How to initialize multimodal training: pretrained backbones only, from unimodal ckpts, or from mm_ckpt.")
	
	# - Model training options
	parser.add_argument('--run_eval_on_start', dest='run_eval_on_start', action='store_true',help='Run model evaluation on start for debug (default=false)')	
	parser.set_defaults(run_eval_on_start=False)
	parser.add_argument('--run_eval_on_start_manual', dest='run_eval_on_start_manual', action='store_true',help='Run model evaluation on start for debug (default=false)')	
	parser.set_defaults(run_eval_on_start_manual=False)
	parser.add_argument('-logging_steps', '--logging_steps', dest='logging_steps', required=False, type=int, default=1, action='store',help='NUmber of logging steps (default=1)')
	parser.add_argument('--run_eval_on_step', dest='run_eval_on_step', action='store_true',help='Run model evaluation after each step (default=false)')	
	parser.set_defaults(run_eval_on_step=False)
	parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', dest='gradient_accumulation_steps', required=False, type=int, default=1, action='store',help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass (default=1)')

	#parser.add_argument('-ngpu', '--ngpu', dest='ngpu', required=False, type=int, default=1, action='store', help='Number of gpus used for the run. Needed to compute the global number of training steps (default=1)')	
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=1, action='store', help='Number of epochs used in network training (default=100)')	
	#parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='adamw', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-lr_scheduler', '--lr_scheduler', dest='lr_scheduler', required=False, type=str, default='cosine', action='store',help='Learning rate scheduler used {constant, linear, cosine, cosine_with_min_lr} (default=cosine)')
	parser.add_argument('-lr', '--lr', dest='lr', required=False, type=float, default=5e-5, action='store',help='Learning rate (default=5e-5)')
	#parser.add_argument('-min_lr', '--min_lr', dest='min_lr', required=False, type=float, default=1e-6, action='store',help='Learning rate min used in cosine_with_min_lr (default=1.e-6)')
	parser.add_argument('-warmup_ratio', '--warmup_ratio', dest='warmup_ratio', required=False, type=float, default=0.2, action='store',help='Warmup ratio par (default=0.2)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=8, action='store',help='Batch size used in training (default=8)')
	parser.add_argument('-batch_size_eval', '--batch_size_eval', dest='batch_size_eval', required=False, type=int, default=None, action='store',help='Batch size used for evaluation. If None set equal to train batch size (default=None)')
	
	parser.add_argument('--drop_last', dest='drop_last', action='store_true',help='Drop last incomplete batch (default=false)')	
	parser.set_defaults(drop_last=False)
	
	parser.add_argument('-weight_decay','--weight_decay', dest='weight_decay', type=float, default=0.0, help='AdamW weight decay (default=0.0)')
	parser.add_argument('--head_dropout', dest='head_dropout', type=float, default=0.0, help='Dropout prob before classifier heads (default=0.0)')
	parser.add_argument('--proj_dropout', dest='proj_dropout', type=float, default=0.0, help='Dropout prob applied to per-timestep projected features before Moirai (imgfeatts).')
	parser.add_argument('--enc_hidden_dropout', type=float, default=None, help='Override encoder hidden_dropout_prob (if supported by the backbone).')
	parser.add_argument('--enc_attn_dropout', type=float, default=None, help='Override encoder attention_probs_dropout_prob (if supported by the backbone).')
                    
	parser.add_argument('--ddp_find_unused_parameters', dest='ddp_find_unused_parameters', action='store_true', help='Flag passed to DistributedDataParallel when using distributed training (default=false)')	
	parser.set_defaults(ddp_find_unused_parameters=False)
	parser.add_argument('--fp16', dest='fp16', action='store_true', help='Enable fp16 (default=false)')	
	parser.set_defaults(fp16=False)
	parser.add_argument('--bf16', dest='bf16', action='store_true', help='Enable bf16 (default=false)')	
	parser.set_defaults(bf16=False)
	
	parser.add_argument('-metric_for_best_model', '--metric_for_best_model', dest='metric_for_best_model', required=False, type=str, default='tss', action='store', help='Metric used to select the best model (default=eval/tss)')
	parser.add_argument('--compute_best_tss', dest='compute_best_tss', action='store_true', help='Compute TSS best vs threshold in evaluation (default=false)')	
	parser.set_defaults(compute_best_tss=False)
	
	parser.add_argument('--compute_metrics_vs_thr', dest='compute_metrics_vs_thr', action='store_true', help='Compute metrics vs threshold in evaluation (default=false)')	
	parser.set_defaults(compute_metrics_vs_thr=False)
	
	parser.add_argument('-seed', '--seed', dest='seed', required=False, type=int, default=42, action='store',help='Random seed that will be set at the beginning of training (default=42)')
	
	parser.add_argument('--print_all_model_layers', dest='print_all_model_layers', action='store_true', help='Print all model layers for debug (default=false)')	
	parser.set_defaults(print_all_model_layers=False)
	
	# - Imbalanced trainer options
	parser.add_argument("--use_weighted_loss", dest='use_weighted_loss', action="store_true", default=False, help="Use class-weighted loss (CE or focal alpha).")
	parser.add_argument("--use_weighted_sampler", dest='use_weighted_sampler', action="store_true", default=False, help="Use a WeightedRandomSampler for training.")
	parser.add_argument("--sample_weight_from_flareid", dest='sample_weight_from_flareid', action="store_true", default=False, help="Compute sample weights from flare id (mostly used for binary class).")
	parser.add_argument("--weight_compute_mode", dest='weight_compute_mode', type=str, choices=["balanced", "inverse", "inverse_v2"], default="balanced", help="How to compute class weights")
	parser.add_argument("--sample_weight_compute_mode", dest='sample_weight_compute_mode', type=str, choices=["balanced", "inverse", "inverse_v2"], default="balanced", help="How to compute sample weights")
	parser.add_argument('--normalize_weights', dest='normalize_weights', action='store_true', help="Enable normalization of class weights.")
	parser.add_argument('--no_normalize_weights', dest='normalize_weights', action='store_false', help="Disable normalization of class weights.")
	parser.set_defaults(normalize_weights=True)
	
	parser.add_argument("--loss_type", dest='loss_type', type=str, choices=["ce", "focal", "sol"], default="ce", help="Loss type: standard cross-entropy, focal loss or solar custom loss.")
	parser.add_argument("--focal_gamma", dest='focal_gamma', type=float, default=2.0, help="Focal loss gamma (focusing parameter).")
	parser.add_argument("--set_focal_alpha_to_mild_estimate", dest='set_focal_alpha_to_mild_estimate', action="store_true", default=False, help="Set focal alpha to mild estimate, otherwise to class_weights.")
	
	parser.add_argument('-sol_score', '--sol_score', dest='sol_score', choices=["accuracy", "precision", "recall", "specificity", "f1", "tss", "csi", "hss1", "hss2"], required=False, type=str, default='tss', action='store', help='Solar score used (default=tss)')
	parser.add_argument('-sol_distribution', '--sol_distribution', dest='sol_distribution', choices=["uniform", "cosine"], required=False, type=str, default='uniform', action='store', help='Solar score distribution used (default=uniform)')
	parser.add_argument('-sol_mode', '--sol_mode', dest='sol_mode', choices=["weighted", "average"], required=False, type=str, default='average', action='store', help='Solar score averaging mode used (default=average)')
	parser.add_argument("--sol_add_constant", dest='sol_add_constant', action="store_true", default=False, help="Add constant (+1) to solar loss (default=false).")
		
	parser.add_argument('-ordinal_thresholds', '--ordinal_thresholds', dest='ordinal_thresholds', required=False, type=list_of_floats, default=None, action='store', help='Sigmoid thresholds (e.g. [0.5,0.5,0.5]) for the K-1 flare classes. If None, 0.5 per class. (default=None)')
		
	parser.add_argument("--compute_train_metrics", dest='compute_train_metrics', action="store_true", default=False, help="Compute and log train metrics during training.")
	parser.add_argument("--clear_eval_cache", dest='clear_eval_cache', action="store_true", default=False, help="Clear cache allocator for eval during training.")
		
	# - Run options
	parser.add_argument('-device', '--device', dest='device', required=False, type=str, default="cuda:0", action='store', help='Device identifier')
	parser.add_argument('-runname', '--runname', dest='runname', required=False, type=str, default="sfforecast", action='store', help='Run name')
	parser.add_argument('--verbose', dest='verbose', action='store_true',help='Enable verbose printout (default=false)')	
	parser.set_defaults(verbose=False)
	
	parser.add_argument('--predict', dest='predict', action='store_true', help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)
	parser.add_argument('--test', dest='test', action='store_true', help='Run model test on input data (default=false)')	
	parser.set_defaults(test=False)
	
	parser.add_argument("--report_to", dest='report_to', type=str, default="wandb", help="Report logs/metrics to {wandb, none}")

	parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
	parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", -1)))
	parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", -1)))

	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--pin_memory", type=str, choices=["true","false"], default="false")
	parser.add_argument("--persistent_workers", type=str, choices=["true","false"], default="false")

	# - Output options
	parser.add_argument('-outdir','--outdir', dest='outdir', required=False, default="", type=str, help='Output data dir') 
	#parser.add_argument('--save_model_every_epoch', dest='save_model_every_epoch', action='store_true', help='Save model every epoch (default=false)')	
	#parser.set_defaults(save_model_every_epoch=False)
	parser.add_argument('-max_checkpoints', '--max_checkpoints', dest='max_checkpoints', required=False, type=int, default=2, action='store',help='Max number of saved checkpoints (default=2)')
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, default="classifier_results.json", type=str, help='Output file with saved inference results') 
	parser.add_argument("--save_metric_curves", action="store_true", help="If set, save precision/recall/F1/TSS/HSS/MCC/ApSS vs threshold to CSV for eval and test.")
	
	# - Parse arguments
	#   NB: Accept unknown args so launchers can't break
	#args = parser.parse_args()	
	args, _unknown = parser.parse_known_args()
	
	return args		
	
	
#####################
##   LOAD MODELS   ##
#####################
def load_ordinal_image_model(
	args,
	num_classes=4
):
	"""
		Loads an HF image classifier, replaces its classifier with CoralOrdinalHead,
		and configures it for ordinal training.
	"""
    
	# - We will emit K-1 logits for cumulative tasks
	#   Override id2label & label2id
	num_labels = num_classes - 1  # num_classes=4
	id2label= {0: ">=C", 1: ">=M", 2: ">=X"} if num_classes == 4 else {i: f">=class{i+1}" for i in range(num_labels)}
	label2id= {">=C": 0, ">=M": 1, ">=X": 2} if num_classes == 4 else {f">=class{i+1}": i for i in range(num_labels)}
	
	print("--> Ordinal id2label")
	print(id2label)
	print("--> Ordinal label2id")
	print(label2id)
	
	# - Define model
	model = AutoModelForImageClassification.from_pretrained(
		args.model,
		problem_type="multi_label_classification",   # BCEWithLogitsLoss inside HF
		num_labels=num_labels,
		id2label=id2label,
		label2id=label2id,
	)

	# - Get input feature size from the existing classifier
	if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
		in_features = model.classifier.in_features
		model.classifier = CoralOrdinalHead(in_features, num_classes=num_classes)
	else:
		raise RuntimeError("Could not locate a linear classification layer named classifier to replace.")

	return model
	

def load_image_model_vit(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load image model & processor (ViT loader version) """

	if inference_mode:
		# - Load model for inference
		model= ViTForImageClassification.from_pretrained(args.model)
		
		# Ensure head matches the checkpoint if it was trained with Dropout+Linear
		try:
			state = load_state_dict_any(args.model)  # works with checkpoint dir or .bin/.safetensors
			needs_seq_head = any(k.startswith("classifier.1.") for k in state.keys())
			has_plain_linear = hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear)
			if needs_seq_head and has_plain_linear:
				p = getattr(getattr(model, "config", object()), "head_dropout", args.head_dropout)
				maybe_wrap_classifier_with_dropout(model, p)
				# reload so classifier.* weights map correctly
				model.load_state_dict(state, strict=False)
		except Exception as e:
			logger.warning(f"Classifier head alignment skipped (non-fatal): {e}")
		
		model.eval()
		
	else:
		# - Load model for training
		if args.binary:
			config= ViTConfig.from_pretrained(
				args.model,
				problem_type=None, 
				num_labels=1
			)	
		else:
			config= ViTConfig.from_pretrained(
				args.model,
				problem_type="single_label_classification", 
				id2label=id2label, 
				label2id=label2id,
				num_labels=num_labels
			)
		
		model= ViTForImageClassification.from_pretrained(
			args.model,
			config=config
		)
		
		# - Replace the head with 1 logit for binary class
		if args.binary:
			in_features = model.classifier.in_features
			model.classifier = torch.nn.Linear(in_features, 1)
			model.config.num_labels = 1  # avoid confusion; we’ll provide our own loss
			model.config.problem_type = None  # don't let HF pick MSE; we handle loss in Trainer
					
		# - Add dropout layer in architecture and config?
		maybe_wrap_classifier_with_dropout(
			model, args.head_dropout, num_out=(1 if args.binary else num_labels)
		)
		if hasattr(model, "config"):
			setattr(model.config, "head_dropout", float(args.head_dropout))	
			
		#if args.head_dropout > 0.0 and hasattr(model, "classifier"):
		#	logger.info("Adding dropout layer in classifier head ...")
		#	in_features = model.classifier.in_features
		#	out_features = 1 if args.binary else num_labels
		#	model.classifier = torch.nn.Sequential(
		#		torch.nn.Dropout(p=args.head_dropout),
		#		torch.nn.Linear(in_features, out_features)
		#	)	
		
				
	# - Load processor	
	image_processor = ViTImageProcessor.from_pretrained(args.model)
	
	return model, image_processor
		

def load_image_model_auto(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load image model & processor (AutoModelForImageClassification loader version) """

	if inference_mode:
		# - Load model for inference
		model = AutoModelForImageClassification.from_pretrained(args.model)
		
		# - Ensure head matches the checkpoint if it was trained with Dropout+Linear
		try:
			state = load_state_dict_any(args.model)  # works with checkpoint dir or .bin/.safetensors
			needs_seq_head = any(k.startswith("classifier.1.") for k in state.keys())
			has_plain_linear = hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear)
			if needs_seq_head and has_plain_linear:
				p = getattr(getattr(model, "config", object()), "head_dropout", args.head_dropout)
				maybe_wrap_classifier_with_dropout(model, p)
				# reload so classifier.* weights map correctly
				model.load_state_dict(state, strict=False)
		except Exception as e:
			logger.warning(f"Classifier head alignment skipped (non-fatal): {e}")
		
		model.eval()
		
	else:
		# - Load model for training
		if args.ordinal:
			# - Load ordinal-head model
			model= load_ordinal_image_model(args, nclasses)
		
		else:
			# - Load standard-head model
			if args.binary:		
				# - Load standard tmp model
				model = AutoModelForImageClassification.from_pretrained(args.model, num_labels=2)  # temp
				
				# - Replace the head with 1 logit
				in_features = model.classifier.in_features
				model.classifier = torch.nn.Linear(in_features, 1)
				model.config.num_labels = 1  # avoid confusion; we'll provide our own loss
				model.config.problem_type = None  # don't let HF pick MSE; we handle loss in Trainer	
					
			else:
				# - Load standard model
				model = AutoModelForImageClassification.from_pretrained(
					args.model, 
					problem_type="single_label_classification", 
					id2label=id2label, 
					label2id=label2id,
					num_labels=num_labels
				)
			
			# - Add dropout in architecture & config?	
			maybe_wrap_classifier_with_dropout(
				model, args.head_dropout, num_out=(1 if args.binary else num_labels)
			)
			if hasattr(model, "config"):
				setattr(model.config, "head_dropout", float(args.head_dropout))
			#if args.head_dropout > 0.0 and hasattr(model, "classifier"):
			#	logger.info("Adding dropout layer in classifier head ...")
			#	in_features = model.classifier.in_features
			#	out_features = 1 if args.binary else num_labels
			#	model.classifier = torch.nn.Sequential(
			#		torch.nn.Dropout(p=args.head_dropout),
			#		torch.nn.Linear(in_features, out_features)
			#	)
		
	# - Load processor	
	image_processor = AutoImageProcessor.from_pretrained(args.model)
		
	return model, image_processor	


def load_image_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load image model & processor """

	#===================================
	#==     VIT-LOADER
	#===================================
	# - Load model
	if args.vitloader:
		model, image_processor= load_image_model_vit(
			args,	
			id2label,
			label2id,
			num_labels,
			nclasses,
			inference_mode
		)
	
	#===================================
	#==     AUTOMODEL
	#===================================
	else:
		model, image_processor= load_image_model_auto(
			args,	
			id2label,
			label2id,
			num_labels,
			nclasses,
			inference_mode
		)
		
	return model, image_processor	
	
			
def load_videomae_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False,
	ckpt_override=""
):
	""" Load video model & processor """
	
	if inference_mode:
		# - Load model for inference
		logger.info("Loading VideoMAE for inference ...")
		model = VideoMAEForVideoClassification.from_pretrained(args.model)
		
		# - Ensure head matches the checkpoint if it was trained with Dropout+Linear
		try:
			ckpt = ckpt_override if ckpt_override else args.model
			logger.info(f"Loading checkpoint {ckpt} in VideoMAE ...")
			state = load_state_dict_any(ckpt) # works with checkpoint dir or .bin/.safetensors
			needs_seq_head = any(k.startswith("classifier.1.") for k in state.keys())
			has_plain_linear = hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear)
			logger.info(f"needs_seq_head? {needs_seq_head}, has_plain_linear? {has_plain_linear}")
			
			if needs_seq_head and has_plain_linear:
				logger.info("Check if wrapping classifier with dropout ...")
				p = getattr(getattr(model, "config", object()), "head_dropout", args.head_dropout)
				maybe_wrap_classifier_with_dropout(model, p)
				# reload so classifier.* weights map correctly
				logger.info("Reload model weights so classifier.* weights map correctly ...")
				model.load_state_dict(state, strict=False)
		except Exception as e:
			logger.warning(f"Classifier head alignment skipped (non-fatal): {e}")
			
		model.eval()
		
	else:
		# - Load model for training
		try:
			if args.binary:
				logger.info("Loading VideMAE for training (num_labels=1) ...")
				model = VideoMAEForVideoClassification.from_pretrained(args.model, num_labels=1) # tmp
				in_features = model.classifier.in_features
				model.classifier = torch.nn.Linear(in_features, 1)
				model.config.num_labels = 1
				model.config.problem_type = None
		
			else:
				logger.info(f"Loading VideMAE for training (num_labels={num_labels}) ...")
				model = VideoMAEForVideoClassification.from_pretrained(
					args.model,
					problem_type="single_label_classification", 
					id2label=id2label, 
					label2id=label2id,
					num_labels=num_labels,
					#attn_implementation="sdpa", # "flash_attention_2"
					#torch_dtype=torch.float16,  # "auto"
					#config=cfg,
				)
		except Exception as e:
			logger.warning(f"Failed to load VideoMAE (err={str(e)}), trying alternative method ...")
			model = VideoMAEForVideoClassification.from_pretrained(args.model)
			
			head_numout= 1 if args.binary else num_labels
			in_features = model.classifier.in_features
			model.classifier = torch.nn.Linear(in_features, head_numout)
			model.config.num_labels = head_numout
			model.config.problem_type = None
	
		# - Add dropout layer in architecture and config?
		maybe_wrap_classifier_with_dropout(
			model, args.head_dropout, num_out=(1 if args.binary else num_labels)
		)
		if hasattr(model, "config"):
			setattr(model.config, "head_dropout", float(args.head_dropout))	
		#if args.head_dropout > 0.0 and hasattr(model, "classifier"):
		#	logger.info("Adding dropout layer in classifier head ...")
		#	in_features = model.classifier.in_features
		#	out_features = model.config.num_labels
		#	model.classifier = torch.nn.Sequential(
		#		torch.nn.Dropout(p=args.head_dropout),
		#		torch.nn.Linear(in_features, out_features)
		#	)
	
	print("== VIDEO MAE CONFIG ==")
	print(model.config)
		
	# - Load processor
	image_processor = VideoMAEImageProcessor.from_pretrained(args.model)
	
	# - Check head initialization
	if args.check_videomae_head and not inference_mode:
		logger.info("Checking VideoMAE head weights ...")
		suspicious = check_head_initialization(model)
		if suspicious:
			logger.info("VideoMAE model head has suspicious initialization values, will re-initialize them ...")
			safe_reinit_head(model)

			suspicious= check_head_initialization(model)
			if suspicious:
				logger.warning("After re-initialization, the VideoMAE head weights still are detected as suspicious ...")
			else:
				logger.info("After re-initialization, the VideoMAE head weights look numerically safe to start the training ...")

	return model, image_processor
		
def load_imgfeatts_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load img feat TS model & processor """
		
	num_out= (1 if args.binary else num_labels)
	print("num_out")
	print(num_out)
	
	model = ImageFeatTSClassifier(
		image_model_name=args.model_ts_img_backbone,
		moirai_pretrained_name=args.model_ts_backbone,
		num_labels=num_out,
		proj_dim=args.proj_dim,
		patching_mode=args.ts_patching_mode,
		freeze_backbone=args.ts_freeze_backbone,
		max_freeze_layer_id=args.ts_max_freeze_layer_id,
		freeze_img_backbone=args.freeze_backbone,
		max_img_freeze_layer_id=args.max_freeze_layer_id,
		head_dropout=args.head_dropout,
		proj_dropout=args.proj_dropout,
		encoder_hidden_dropout=args.enc_hidden_dropout,
		encoder_attn_dropout=args.enc_attn_dropout,
	)
	
	# - Add head_dropout option in config
	if hasattr(model, "config"):
		setattr(model.config, "head_dropout", float(args.head_dropout))
		
	# - Get image processor
	image_processor = model.image_processor
	
	# - Ordinal variant (NOT IMPLEMENTED)
	if args.ordinal:
		raise ValueError("Ordinal head not yet implemented for time series data!")
	
	# - Inference?
	if inference_mode:
		# - Patch the scaler exactly like in training (prevents SqrtBackward in-place issues)
		patch_scaler_instance(model.backbone)
			
		# - Load trained checkpoint (weights + config)
		ckpt = args.model  # can be a file OR a checkpoint dir
		logger.info(f"Loading weights from path {args.model} ...")
		state = load_state_dict_any(ckpt)
		missing, unexpected = model.load_state_dict(state, strict=False)
		if missing or unexpected:
			print(f"[load_imgfeatts_model] load_state_dict -> missing: {missing[:6]} ... | unexpected: {unexpected[:6]} ...")
		
		model.eval()
				
	return model, image_processor
		
		
def load_video_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False,
	ckpt_override: str = ""
):
	""" Load video model & processor """
	
	if args.video_model=="videomae":
		return load_videomae_model(
			args=args,
			id2label=id2label,
			label2id=label2id,
			num_labels=num_labels,
			nclasses=nclasses,
			inference_mode=inference_mode,
			ckpt_override= ckpt_override,
		)
	elif args.video_model=="imgfeatts":
		return load_imgfeatts_model(
			args=args,
			id2label=id2label,
			label2id=label2id,
			num_labels=num_labels,
			nclasses=nclasses,
			inference_mode=inference_mode
		)
	else:
		raise ValueError("Invalid/unsupported video_model specified!")
	
	
def load_ts_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False,
	ckpt_override=""
):
	"""Load Moirai model for time-series classification (Trainer-compatible)."""

	# - Create model (lazy head will be created on first forward)
	num_out= (1 if args.binary else num_labels)
	print("num_out")
	print(num_out)
	
	model = MoiraiForSequenceClassification(
		pretrained_name=args.model_ts_backbone,
		num_labels=num_out,
		freeze_backbone=args.ts_freeze_backbone,
		max_freeze_layer_id=args.ts_max_freeze_layer_id,
		patching_mode=args.ts_patching_mode
	)
	
	# keep config in sync for your Trainer / metrics
	cfg = model.config
	if hasattr(cfg, "update"):
		cfg.update({"num_labels": num_out, "id2label": id2label, "label2id": label2id})
	else:
		cfg.num_labels = num_out
		cfg.id2label  = id2label
		cfg.label2id  = label2id
	
	print("--> cfg")
	print(cfg)
	print("--> model.config")
	print(model.config)
	print(model.config.to_json_string())
	         
	# - Ordinal variant (NOT IMPLEMENTED)
	if args.ordinal:
		raise ValueError("Ordinal head not yet implemented for time series data!")
 
	# - Add dropout in architecture & config (applies to train & inference; dropout is inactive in eval)?
	maybe_wrap_classifier_with_dropout(
		model, args.head_dropout, num_out=model.config.num_labels
	)	
	if hasattr(model, "config"):
		setattr(model.config, "head_dropout", float(args.head_dropout))	
	#if not inference_mode and args.head_dropout > 0.0:
	#	hidden_size = model.config.hidden_size
	#	out_features = model.config.num_labels
	#	logger.info(f"Adding dropout {args.head_dropout} before classifier head ...")
	#	model.classifier = nn.Sequential(
	#		torch.nn.Dropout(p=args.head_dropout),
	#		torch.nn.Linear(hidden_size, out_features)
	#	)
			
  # - Inference?
	if inference_mode:
	  # - Load trained checkpoint (weights + config)
		ckpt = ckpt_override if ckpt_override else args.model # can be a file OR a checkpoint dir
		
		logger.info(f"Loading weights from path {ckpt} ...")
		state = load_state_dict_any(ckpt)
		
		# - If checkpoint expects Sequential(Dropout, Linear) but we currently have Linear, fix head first
		needs_seq_head = any(k.startswith("classifier.1.") for k in state.keys())
		has_plain_linear = hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear)
		if needs_seq_head and has_plain_linear:
			p = getattr(getattr(model, "config", object()), "head_dropout", args.head_dropout)
			maybe_wrap_classifier_with_dropout(model, p)
		
		missing, unexpected = model.load_state_dict(state, strict=False)
		if missing or unexpected:
			print(f"[load_ts_model] load_state_dict -> missing: {missing[:6]} ... | unexpected: {unexpected[:6]} ...")
		
		model.eval()
		
	# - No processor needed for TS (your TSDataCollator handles it)
	ts_processor = None

	return model, ts_processor
	

def load_multimodal_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load multimodal video-ts model """
	
	num_out = (1 if args.binary else num_labels)

	# Decide init strategy
	mm_init = getattr(args, "mm_init", "unimodal")
	mm_ckpt = getattr(args, "mm_ckpt", "")
	video_ckpt = getattr(args, "video_ckpt", "")
	ts_ckpt = getattr(args, "ts_ckpt", "")

	# -----------------------
	# 1) Build / init branches
	# -----------------------
	if mm_init == "pretrained":
		# Use pretrained weights only (no branch ckpts)
		logger.info(f"[pretrained init] - Loading video model (num_labels={num_labels}, nclasses={nclasses}, video_inference_mode=False) ...")
		video_model, image_processor = load_video_model(
			args=args,
			id2label=id2label,
			label2id=label2id,
			num_labels=num_labels,
			nclasses=nclasses,
			inference_mode=False,
			ckpt_override=""
		)
		
		logger.info(f"[pretrained init] - Loading ts model (num_labels={num_labels}, nclasses={nclasses}, ts_inference_mode=False) ...")
		ts_model, _tp = load_ts_model(
			args=args,
			id2label=id2label,
			label2id=label2id,
			num_labels=num_labels,
			nclasses=nclasses,
			inference_mode=False,
			ckpt_override=""
		)

	elif mm_init == "unimodal":
		# Init from separately trained unimodal ckpts (if provided), otherwise pretrained
		video_inference_mode= (True if video_ckpt else False)
		logger.info(f"[unimodal init] - Loading video model (num_labels={num_labels}, nclasses={nclasses}, video_inference_mode=video_inference_mode) ...")
		video_model, image_processor = load_video_model(
			args=args,
			id2label=id2label,
			label2id=label2id,
			num_labels=num_labels,
			nclasses=nclasses,
			inference_mode=video_inference_mode,
			ckpt_override=video_ckpt
		)
		if video_inference_mode and not inference_mode:
			video_model.train()
		
		ts_inference_mode= (True if ts_ckpt else False)
		logger.info(f"[unimodal init] - Loading ts model (num_labels={num_labels}, nclasses={nclasses}, ts_inference_mode=ts_inference_mode) ...")
		ts_model, _tp = load_ts_model(
			args=args,
			id2label=id2label,
			label2id=label2id,
			num_labels=num_labels,
			nclasses=nclasses,
			inference_mode=(True if ts_ckpt else False),
			ckpt_override=ts_ckpt
		)
		if ts_inference_mode and not inference_mode:
			ts_model.train()

	elif mm_init == "multimodal":
		# For multimodal ckpt init we still need a *skeleton* model.
		# Build branches from pretrained or from branch ckpts if provided
		video_inference_mode= (True if video_ckpt else False)
		logger.info(f"[multimodal init] - Loading video model (num_labels={num_labels}, nclasses={nclasses}, video_inference_mode=video_inference_mode) ...")
		video_model, image_processor = load_video_model(
			args=args,
			id2label=id2label,
			label2id=label2id,
			num_labels=num_labels,
			nclasses=nclasses,
			inference_mode=video_inference_mode,
			ckpt_override=video_ckpt
		)
		
		ts_inference_mode= (True if ts_ckpt else False)
		logger.info(f"[multimodal init] - Loading ts model (num_labels={num_labels}, nclasses={nclasses}, ts_inference_mode=ts_inference_mode) ...")
		ts_model, _tp = load_ts_model(
			args=args,
			id2label=id2label,
			label2id=label2id,
			num_labels=num_labels,
			nclasses=nclasses,
			inference_mode=ts_inference_mode,
			ckpt_override=ts_ckpt
		)
	else:
		raise ValueError(f"Invalid mm_init={mm_init}")

	# -----------------------
	# 2) Build fusion model
	# -----------------------
	logger.info(f"Building MultimodalConcatMLP model (num_labels={num_out}, hidden_dim={args.mm_hidden_dim}, dropout={args.head_dropout}) ...")
	model = MultimodalConcatMLP(
		video_model=video_model,
		ts_model=ts_model,
		num_labels=num_out,
		hidden_dim=args.mm_hidden_dim,
		dropout=args.head_dropout,
	)

	# -----------------------
	# 3) If mm_ckpt is provided (or init==multimodal), load fusion checkpoint
	# -----------------------
	# This loads the whole multimodal state dict (proj/head + possibly branches).
	# You can control strictness depending on whether ckpt contains full branches.
	ckpt_to_load = ""
	if inference_mode and mm_ckpt:
		ckpt_to_load = mm_ckpt
	elif (mm_init == "multimodal") and mm_ckpt:
		ckpt_to_load = mm_ckpt

	if ckpt_to_load:
		logger.info(f"Loading multimodal checkpoint from {ckpt_to_load} ...")
		state = load_state_dict_any(ckpt_to_load)
		missing, unexpected = model.load_state_dict(state, strict=False)
		if missing or unexpected:
			print(f"[load_multimodal_model] load_state_dict -> missing: {missing[:10]} ... | unexpected: {unexpected[:10]} ...")
		model.eval() if inference_mode else None

	# Processor: for multimodal we need the VIDEO processor
	# NB: Use the image_processor returned by load_video_model() rather than allocating a new one with: image_processor = AutoImageProcessor.from_pretrained(args.model)
	
	return model, image_processor
	
			
def load_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses,
	inference_mode=False
):
	""" Load model & processor """
	
	if args.data_modality=="image":
		logger.info("Loading image model & processor (name=%s) ..." % (args.model))
		return load_image_model(args, id2label, label2id, num_labels, nclasses, inference_mode)

	elif args.data_modality=="video":
		logger.info("Loading video model & processor (name=%s) ..." % (args.model))
		return load_video_model(args, id2label, label2id, num_labels, nclasses, inference_mode)
		
	elif args.data_modality=="ts":
		logger.info("Loading ts model & processor (name=%s) ..." % (args.model_ts_backbone))
		return load_ts_model(args, id2label, label2id, num_labels, nclasses, inference_mode)
		
	elif args.data_modality=="multimodal":
		logger.info(f"Loading multimodal model (ConcatMLP: video={args.model}, ts={args.model_ts_backbone}) ...")
		return load_multimodal_model(args, id2label, label2id, num_labels, nclasses, inference_mode)
		
	else:
		raise ValueError(f"Data modality {args.data_modality} not supported!")


def freeze_model_old(model, args):
	""" Freeze certain part of the model """
	
	# - Set encoder model name
	if args.data_modality=="image":	
		encoder_name= "vision_model.encoder"
		layer_search_pattern= "layers"
		model_base= model.base_model
		
	elif args.data_modality=="video":
		if args.video_model=="imgfeatts":
			return model
			
		encoder_name= "encoder"
		layer_search_pattern= "layer"
		model_base= model.base_model
		
	elif args.data_modality=="multimodal":	# Nothing to be done (freezing done internally) 
		return model
		
	else: # Nothing to be done
		encoder_name= "encoder"
		layer_search_pattern= "layer"
		model_base= model.backbone
		return model
	
	# - Freeze layers
	logger.info("Freezing model base layers ...")
	#for name, param in model.base_model.named_parameters():
	for name, param in model_base.named_parameters():	
		if name.startswith(encoder_name):
			layer_index= extract_layer_id(name, layer_search_pattern)
			if args.max_freeze_layer_id==-1 or (args.max_freeze_layer_id>=0 and layer_index!=-1 and layer_index<args.max_freeze_layer_id):
				param.requires_grad = False
		
	return model


def freeze_model(model, args):
	""" Freeze certain part of the model """

	# -----------------------------------
	# HELPERS
	# -----------------------------------
	def freeze_named_layers(module, encoder_name, layer_search_pattern, max_freeze_layer_id):
		if module is None:
			return
		for name, param in module.named_parameters():
			if name.startswith(encoder_name):
				layer_index = extract_layer_id(name, layer_search_pattern)
				if max_freeze_layer_id == -1 or (layer_index != -1 and layer_index < max_freeze_layer_id):
					logger.debug(f"Freezing layer {name} ...")
					param.requires_grad = False
					
	def freeze_all_params(module):
		if module is None:
			return
		for _, p in module.named_parameters():
			p.requires_grad = False

	def unfreeze_encoder_layers_by_id(module, encoder_name, layer_search_pattern, max_unfreeze_from_layer_id):
		"""
		Unfreezes encoder layers whose index >= max_unfreeze_from_layer_id.
		Example: if max_unfreeze_from_layer_id=8, unfreeze layers 8..end.
		Use -1 to keep everything frozen.
		"""
		if module is None:
			return
		if max_unfreeze_from_layer_id < 0:
			return

		for name, p in module.named_parameters():
			if not name.startswith(encoder_name):
				continue
			layer_index = extract_layer_id(name, layer_search_pattern)
			if layer_index != -1 and layer_index >= max_unfreeze_from_layer_id:
				p.requires_grad = True

	# -----------------------------
	# IMAGE
	# -----------------------------
	if args.data_modality == "image":
		if args.freeze_backbone:
			logger.info("Freezing Image encoder ...")
			model_base = getattr(model, "base_model", None)
			freeze_named_layers(
				model_base,
				encoder_name="vision_model.encoder",
				layer_search_pattern="layers",
				max_freeze_layer_id=args.max_freeze_layer_id
			)
		return model
		
	# -----------------------------
	# VIDEO
	# -----------------------------
	if args.data_modality == "video":
		if args.video_model == "imgfeatts":
			# Freeze IMAGE branch of ImageFeatTSClassifier
			if args.freeze_backbone:
				logger.info("Freezing imgfeatts IMAGE encoder ...")
				img_base = None
				img_parent = getattr(model, "image_enc", None)
				if img_parent is not None:
					img_base = getattr(img_parent, "encoder", None)
				freeze_named_layers(
					img_base,
					encoder_name="encoder",
					layer_search_pattern="layers",
					max_freeze_layer_id=args.max_freeze_layer_id
				)

			# Freeze TS(Moirai) branch of ImageFeatTSClassifier
			if args.ts_freeze_backbone:
				logger.info("Freezing imgfeatts TS encoder ...")
				ts_base = getattr(model, "backbone", None)
				freeze_named_layers(
					ts_base,
					encoder_name="encoder",
					layer_search_pattern="layers",
					max_freeze_layer_id=args.ts_max_freeze_layer_id
				)

			return model
			
		elif args.video_model == "videomae":
			if args.freeze_backbone:
				logger.info("Freezing Video encoder ...")	
				model_base = getattr(model, "base_model", None)
				freeze_named_layers(
					model_base,
					encoder_name="encoder",
					layer_search_pattern="layer",
					max_freeze_layer_id=args.max_freeze_layer_id
				)
			return model
			
		else:
			logger.warning(f"Video model {args.video_model} not recognized, returning same model ...")	
			return model
		
	# -----------------------------
	# TS
	# -----------------------------
	if args.data_modality == "ts":
		if args.ts_freeze_backbone:
			logger.info("Freezing TS encoder ...")	
			ts_base = getattr(model, "backbone", None)
			freeze_named_layers(
				ts_base,
				encoder_name="encoder",
				layer_search_pattern="layers",
				max_freeze_layer_id=args.ts_max_freeze_layer_id
			)
		return model
		
	# -----------------------------
	# MULTIMODAL
	# -----------------------------
	if args.data_modality == "multimodal":
		# Freeze VIDEO branch
		video_parent = getattr(model, "video_model", None)
		video_base = getattr(video_parent, "base_model", None) if video_parent is not None else None
		if video_base is None and video_parent is not None:
			video_base = getattr(video_parent, "videomae", None)
		if video_base is None:
			video_base = video_parent
	
		if args.freeze_backbone:
			logger.info("Freezing entire Video backbone ...")
			freeze_all_params(video_base)

			# Optional partial unfreeze: interpret args.max_freeze_layer_id as "freeze up to"
			# Here: keep layers < max_freeze_layer_id frozen, unfreeze >= max_freeze_layer_id
			if args.max_freeze_layer_id >= 0:
				logger.info(f"Unfreezing Video encoder layers >= {args.max_freeze_layer_id} ...")
				unfreeze_encoder_layers_by_id(
					video_base,
					encoder_name="encoder",
					layer_search_pattern="layer",
					max_unfreeze_from_layer_id=args.max_freeze_layer_id
				)
		
		#if args.freeze_backbone:
		#	logger.info("Freezing Video encoder ...")	
		#	video_parent = getattr(model, "video_model", None)
		#	video_base = getattr(video_parent, "base_model", None) if video_parent is not None else None
		#	if video_base is None and video_parent is not None:
		#		video_base = getattr(video_parent, "videomae", None)
		#	if video_base is None:
		#		video_base = video_parent
		#	freeze_named_layers(
		#		video_base,
		#		encoder_name="encoder",
		#		layer_search_pattern="layer",
		#		max_freeze_layer_id=args.max_freeze_layer_id
		#	)

		# Freeze TS branch
		ts_base = getattr(model, "ts_backbone", None)
		if ts_base is None:
			ts_model = getattr(model, "ts_model", None)
			ts_base = getattr(ts_model, "backbone", None) if ts_model is not None else None

		if args.ts_freeze_backbone:
			logger.info("Freezing entire TS backbone ...")
			freeze_all_params(ts_base)

			if args.ts_max_freeze_layer_id >= 0:
				logger.info(f"Unfreezing TS encoder layers >= {args.ts_max_freeze_layer_id} ...")
				unfreeze_encoder_layers_by_id(
					ts_base,
					encoder_name="encoder",
					layer_search_pattern="layers",
					max_unfreeze_from_layer_id=args.ts_max_freeze_layer_id
				)
			
		#if args.ts_freeze_backbone:
		#	logger.info("Freezing TS encoder ...")	
		#	ts_base = getattr(model, "ts_backbone", None)
		#	if ts_base is None:
		#		ts_model = getattr(model, "ts_model", None)
		#		ts_base = getattr(ts_model, "backbone", None) if ts_model is not None else None
		#	freeze_named_layers(
		#		ts_base,
		#		encoder_name="encoder",
		#		layer_search_pattern="layers",
		#		max_freeze_layer_id=args.ts_max_freeze_layer_id
		#	)
		
		return model

	return model


def print_model(model, args, only_frozen=True, only_trainable=False, max_lines=1000):
	"""
	Print model parameters and whether they are frozen.
	Works for: image, video, ts, multimodal (and video_model=imgfeatts).
	"""

	if only_frozen and only_trainable:
		raise ValueError("print_model: choose only one of only_frozen or only_trainable")

	def _print_params(module, prefix, only_frozen=True, only_trainable=False, max_lines=1000):
		if module is None:
			logger.warning(f"[print_model] {prefix}: module is None")
			return 0

		n = 0
		for name, p in module.named_parameters():
			if only_frozen and p.requires_grad:
				continue
			if only_trainable and not p.requires_grad:
				continue

			logger.info(f"--> {prefix}{name}\trequires_grad={p.requires_grad}\tshape={tuple(p.shape)}")
			n += 1

			if max_lines is not None and n >= max_lines:
				logger.info(f"{prefix}... (truncated at {max_lines} lines)")
				break
		return n

	def _get_video_backbone(video_model):
		# HF VideoMAEForVideoClassification usually exposes base_model; fallback to videomae or itself
		if video_model is None:
			return None
		return getattr(video_model, "base_model", getattr(video_model, "videomae", video_model))

	def _get_ts_backbone(ts_model):
		if ts_model is None:
			return None
		return getattr(ts_model, "backbone", ts_model)

	logger.info(f"[print_model] only_frozen={only_frozen}, max_lines={max_lines}")

	# -------------------------
	# Print modality-specific "backbones"
	# -------------------------
	if args.data_modality == "image":
		logger.info("[print_model] IMAGE: base_model parameters")
		n = _print_params(getattr(model, "base_model", None), prefix="base.", only_frozen=only_frozen, only_trainable=only_trainable, max_lines=max_lines)
		if n == 0 and only_frozen:
			logger.info("[print_model] No frozen parameters found in IMAGE base_model.")

	elif args.data_modality == "video":
		if args.video_model == "imgfeatts":
			# ImageFeatTSClassifier: has both image encoder and ts backbone
			logger.info("[print_model] VIDEO(imgfeatts): image encoder parameters")
			_print_params(getattr(model, "image_enc", None), prefix="image_enc.", only_frozen=only_frozen, only_trainable=only_trainable, max_lines=max_lines)

			logger.info("[print_model] VIDEO(imgfeatts): ts backbone parameters")
			_print_params(_get_ts_backbone(getattr(model, "backbone", None)), prefix="ts.", only_frozen=only_frozen, only_trainable=only_trainable, max_lines=max_lines)

		else:
			logger.info("[print_model] VIDEO(videomae): base_model parameters")
			n = _print_params(getattr(model, "base_model", None), prefix="base.", only_frozen=only_frozen, only_trainable=only_trainable, max_lines=max_lines)
			if n == 0 and only_frozen:
				logger.info("[print_model] No frozen parameters found in VIDEO base_model.")

	elif args.data_modality == "ts":
		logger.info("[print_model] TS: backbone parameters")
		n = _print_params(getattr(model, "backbone", None), prefix="ts.", only_frozen=only_frozen, only_trainable=only_trainable, max_lines=max_lines)
		if n == 0 and only_frozen:
			logger.info("[print_model] No frozen parameters found in TS backbone.")

	elif args.data_modality == "multimodal":
		logger.info("[print_model] MULTIMODAL: video backbone parameters")
		vbase = _get_video_backbone(getattr(model, "video_model", None))
		_print_params(vbase, prefix="video.", only_frozen=only_frozen, only_trainable=only_trainable, max_lines=max_lines)

		logger.info("[print_model] MULTIMODAL: ts backbone parameters")
		tbase = getattr(model, "ts_backbone", None)
		if tbase is None:
			# fallback if only ts_model is present
			tbase = _get_ts_backbone(getattr(model, "ts_model", None))
		_print_params(tbase, prefix="ts.", only_frozen=only_frozen, only_trainable=only_trainable, max_lines=max_lines)

	else:
		logger.warning(f"[print_model] Unsupported data_modality={args.data_modality}. Printing full model params.")
		_print_params(model, prefix="", only_frozen=only_frozen, only_trainable=only_trainable, max_lines=max_lines)

	# -----------------------------------------
	#  Print modality-specific NON backbone
	# -----------------------------------------
	logger.info("[print_model] NON-BACKBONE params (heads/proj/etc.)")
	n = 0
	for name, p in model.named_parameters():

		# Skip known backbone prefixes depending on modality
		if args.data_modality == "image":
			if name.startswith("base_model."):
				continue

		elif args.data_modality == "video":
			if args.video_model == "imgfeatts":
				if name.startswith("image_enc.") or name.startswith("backbone."):
					continue
			else:
				if name.startswith("base_model."):
					continue

		elif args.data_modality == "ts":
			if name.startswith("backbone."):
				continue

		elif args.data_modality == "multimodal":
			if name.startswith("video_model.") or name.startswith("ts_model.") or name.startswith("ts_backbone."):
				continue

		# Apply filters
		if only_frozen and p.requires_grad:
			continue
		if only_trainable and not p.requires_grad:
			continue

		logger.info(f"--> head.{name}\trequires_grad={p.requires_grad}\tshape={tuple(p.shape)}")
		n += 1
		if max_lines is not None and n >= max_lines:
			logger.info("head.... (truncated)")
			break

	# -------------------------
	# Optional: print fusion/head params for multimodal
	# -------------------------
	if args.data_modality == "multimodal":
		logger.info("[print_model] MULTIMODAL: fusion head parameters")
		# Print only params outside the two backbones (proj/head) by filtering prefixes
		n = 0
		for name, p in model.named_parameters():
			if name.startswith("video_model.") or name.startswith("ts_model.") or name.startswith("ts_backbone."):
				continue
			if only_frozen and p.requires_grad:
				continue
			if only_trainable and not p.requires_grad:
				continue
					
			logger.info(f"--> fusion.{name}\trequires_grad={p.requires_grad}\tshape={tuple(p.shape)}")
			n += 1
			if max_lines is not None and n >= max_lines:
				logger.info("fusion.... (truncated)")
				break

	# -------------------------
	# Summary counts
	# -------------------------
	total = 0
	frozen = 0
	trainable = 0
	for _, p in model.named_parameters():
		total += p.numel()
		if p.requires_grad:
			trainable += p.numel()
		else:
			frozen += p.numel()

	logger.info(f"[print_model] params: trainable={trainable} / total={total} ({(100.0*trainable/max(1,total)):.2f}%), frozen={frozen} / total={total} ({(100.0*frozen/max(1,total)):.2f}%)")

def print_all_model_params(model):
	""" Print all model parameters """
	for name, param in model.named_parameters():
		print(name, param.requires_grad)

########################
##   LOAD TRANSFORM   ##
########################
def load_video_transform(args, image_processor):
	""" Load video data transform """
	
	# - Retrieve image processor transform parameters
	try:
		size = (image_processor.size["height"], image_processor.size["width"])
	except:
		size= (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])  # VideoMAE does not have height & width inside size
		
	mean = image_processor.image_mean
	std = image_processor.image_std
	
	print("*** Image processor config pars ***")
	print("do_resize? ", (image_processor.do_resize))
	print("size: ", (size))
	print("do_rescale? ", (image_processor.do_rescale))
	print("rescale_factor: ", (image_processor.rescale_factor))
	print("do_normalize? ", (image_processor.do_normalize))
	print("mean: ", (mean))
	print("std: ", (std))
	
	# - Define transforms
	mean= [0.,0.,0.]
	std= [1.,1.,1.]
	sigma_min= 1.0
	sigma_max= 3.0
	ksize= 3.3 * sigma_max
	kernel_size= int(max(ksize, 5)) # in imgaug kernel_size viene calcolato automaticamente dalla sigma così, ma forse si può semplificare a 3x3
	#blur_aug= T.GaussianBlur(kernel_size, sigma=(sigma_min, sigma_max))
	crop_aug= VideoRandomCenterCrop(min_frac=args.min_crop_fract, max_frac=1.0, output_size=None, channels_first_time_dim=True)

	transf_list= []
	if args.add_crop_augm:
		transf_list.append(crop_aug)
		
	transf_list.extend(
		[
			VideoResize(size, interpolation=T.InterpolationMode.BICUBIC),
			VideoFlipping(),
			VideoRotate90(),
			VideoNormalize(mean=mean, std=std)
		]
	)

	#transform_train= T.Compose([
	#	VideoResize(size, interpolation=T.InterpolationMode.BICUBIC),
	#	VideoFlipping(),
	#	VideoRotate90(),
	#	VideoNormalize(mean=mean, std=std),
	#])
		
	transform_train= T.Compose(transf_list)
		
	transform= T.Compose([
		VideoResize(size, interpolation=T.InterpolationMode.BICUBIC),
		VideoNormalize(mean=mean, std=std),
	])
	
	return transform_train, transform
	

def load_image_transform(args, image_processor):
	""" Load image data transform """
		
	# - Retrieve image processor transform parameters
	try:
		size = (image_processor.size["height"], image_processor.size["width"])
	except:
		size= (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])  # VideoMAE does not have height & width inside size
		
	mean = image_processor.image_mean
	std = image_processor.image_std
	
	print("*** Image processor config pars ***")
	print("do_resize? ", (image_processor.do_resize))
	print("size: ", (size))
	print("do_rescale? ", (image_processor.do_rescale))
	print("rescale_factor: ", (image_processor.rescale_factor))
	print("do_normalize? ", (image_processor.do_normalize))
	print("mean: ", (mean))
	print("std: ", (std))
	print("do_convert_rgb? ", (image_processor.do_convert_rgb))
	
	# - Define transforms
	mean= [0.,0.,0.]
	std= [1.,1.,1.]
	sigma_min= 1.0
	sigma_max= 3.0
	ksize= 3.3 * sigma_max
	kernel_size= int(max(ksize, 5)) # in imgaug kernel_size viene calcolato automaticamente dalla sigma così, ma forse si può semplificare a 3x3
	#blur_aug= T.GaussianBlur(kernel_size, sigma=(sigma_min, sigma_max))
	crop_aug= RandomCenterCrop(min_frac=args.min_crop_fract, max_frac=1.0, output_size=None)

	transf_list= []
	if args.add_crop_augm:
		transf_list.append(crop_aug)
		
	transf_list.extend(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			FlippingTransform(),
			Rotate90Transform(),
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)

	transform_train = T.Compose(transf_list)
	
	#transform_train = T.Compose(
	#	[
	#		T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
	#		FlippingTransform(),
	#		Rotate90Transform(),
	#		#T.ToTensor(),
	#		T.Normalize(mean=mean, std=std),
	#	]
	#)
	
	transform = T.Compose(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)
	
	return transform_train, transform
		

######################
##   LOAD DATASET   ##
######################
def load_dataset(
	args, 
	image_processor,
	nclasses,
	id2target
):
	""" Load dataset """
	
	#====================================
	#==   SET OPTIONS
	#====================================
	ts_vars= [str(x.strip()) for x in args.ts_vars.split(',')]
	ts_logstretchs= [bool(int(x.strip())) for x in args.ts_logstretchs.split(',')]
	
	#====================================
	#==   CREATE DATA TRANSFORMS
	#====================================
	# - Load data transforms
	if args.data_modality=="image":
		transform_train, transform_valtest= load_image_transform(args, image_processor)
	elif args.data_modality=="video":
		transform_train, transform_valtest= load_video_transform(args, image_processor)
	elif args.data_modality=="ts":
		#raise ValueError("IMPLEMENT DATA TRANSFORMS FOR TIME SERIES!")
		logger.warning("No transforms are implemented for time-series data ...")
		transform_train= None
		transform_valtest= None
	elif args.data_modality=="multimodal":
		transform_train, transform_valtest= load_video_transform(args, image_processor)
	else:
		raise ValueError(f"Data modality {args.data_modality} not supported!")
		
	#====================================
	#==   CREATE DATASET
	#====================================
	# - Init stuff
	dataset_cv= None
	dataset= None
	nsamples= 0
	nsamples_cv= 0
	DatasetClass= None
	
	if args.data_modality=="image":
		DatasetClass= ImgDataset
	elif args.data_modality=="video":
		DatasetClass= VideoDataset
	elif args.data_modality=="ts":
		DatasetClass= TSDataset
	elif args.data_modality=="multimodal":
		DatasetClass= MultimodalDataset
	else:
		raise ValueError(f"Data modality {args.data_modality} not supported!")
	
	transform= transform_train
	if args.predict or args.test:
		transform= transform_valtest
		
	# - Create train (or test set if args.predict or args.test)
	if args.predict or args.test:
		logger.info("Create dataset for prediction/test ...")
	else:
		logger.info("Create train dataset ...")
	
	if args.data_modality=="image" or args.data_modality=="video":
		dataset= DatasetClass(
			filename=args.datalist,
			transform=transform,
			verbose=args.verbose,
			load_as_gray=args.grayscale,
			apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
			resize=args.resize, resize_size=args.resize_size,
			apply_asinh_stretch=args.asinh_stretch,
			pmin=args.pmin,
			pmax=args.pmax,
			asinh_scale=args.asinh_scale,			
			nclasses=nclasses,
			id2target=id2target,
			multiout=args.multiout,
			multilabel=args.multilabel,
			ordinal=args.ordinal,
		)
		
	elif args.data_modality=="ts":
		dataset= DatasetClass(
			filename=args.datalist,
			transform=transform,
			verbose=args.verbose,
			nclasses=nclasses,
			id2target=id2target,
			multiout=args.multiout,
			multilabel=args.multilabel,
			ordinal=args.ordinal,
			data_vars=ts_vars,
			logstretch_vars=ts_logstretchs,
			npoints=args.ts_npoints
		)
		
	elif args.data_modality=="multimodal":
		dataset = DatasetClass(
			filename=args.datalist,
			video_transform=transform,	# keep None if your BaseVisDataset handles preprocessing internally
			ts_transform=None,
			verbose=args.verbose,
			data_vars=tuple(args.ts_vars.split(",")) if isinstance(args.ts_vars, str) else tuple(args.ts_vars),
			logstretch_vars=tuple([bool(int(x)) for x in args.ts_logstretchs.split(",")]) if isinstance(args.ts_logstretchs, str) else tuple(args.ts_logstretchs),
			npoints=int(args.ts_npoints),
			nclasses=nclasses,
			id2target=id2target,
			multiout=args.multiout,
			multilabel=args.multilabel,
			ordinal=args.ordinal,
			require_matched=args.require_matched,
		)
		
	nsamples= dataset.get_sample_size()
	
	logger.info("#%d entries in dataset ..." % (nsamples))
		
	# - Create validation set?
	if args.datalist_cv!="":
		if args.data_modality=="image" or args.data_modality=="video":
			dataset_cv= DatasetClass(
				filename=args.datalist_cv,
				transform=transform_valtest,
				verbose=args.verbose,
				load_as_gray=args.grayscale,
				apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
				resize=args.resize, resize_size=args.resize_size,
				apply_asinh_stretch=args.asinh_stretch,
				pmin=args.pmin,
				pmax=args.pmax,
				asinh_scale=args.asinh_scale,		
				nclasses=nclasses,
				id2target=id2target,
				multiout=args.multiout,
				multilabel=args.multilabel,
				ordinal=args.ordinal,
			)
		elif args.data_modality=="ts":
			dataset_cv= DatasetClass(
				filename=args.datalist_cv,
				transform=transform_valtest,
				verbose=args.verbose,
				nclasses=nclasses,
				id2target=id2target,
				multiout=args.multiout,
				multilabel=args.multilabel,
				ordinal=args.ordinal,
				data_vars=ts_vars,
				logstretch_vars=ts_logstretchs,
				npoints=args.ts_npoints
			)
		elif args.data_modality=="multimodal":
			dataset_cv = DatasetClass(
				filename=args.datalist_cv,
				video_transform=transform_valtest,	# keep None if your BaseVisDataset handles preprocessing internally
				ts_transform=None,
				verbose=args.verbose,
				data_vars=tuple(args.ts_vars.split(",")) if isinstance(args.ts_vars, str) else tuple(args.ts_vars),
				logstretch_vars=tuple([bool(int(x)) for x in args.ts_logstretchs.split(",")]) if isinstance(args.ts_logstretchs, str) else tuple(args.ts_logstretchs),
				npoints=int(args.ts_npoints),
				nclasses=nclasses,
				id2target=id2target,
				multiout=args.multiout,
				multilabel=args.multilabel,
				ordinal=args.ordinal,
				require_matched=args.require_matched,
			)
		
		nsamples_cv= dataset_cv.get_sample_size()
		logger.info("#%d entries in val dataset ..." % (nsamples_cv))
	
	return dataset, dataset_cv
			
			
############################
##   LOAD TRAINING OPTS   ##
############################
def load_training_opts(args):
	""" Prepare training options """			
			
	# - Set output dir
	output_dir= args.outdir
	if output_dir=="":
		output_dir= os.getcwd()
			
	log_dir= os.path.join(output_dir, "logs/")
	
	# - Set eval & save strategy
	eval_strategy= "no"
	load_best_model_at_end= False
	save_strategy= "no"
	batch_size_eval= args.batch_size if args.batch_size_eval is None else args.batch_size_eval
	if args.datalist_cv!="":
		load_best_model_at_end= True
		if args.run_eval_on_step:
			eval_strategy= "steps"
			save_strategy= "steps"
		else:
			eval_strategy= "epoch"
			save_strategy= "epoch"
			
	# - Set training options
	logger.info("Set model options ...")
	training_opts= transformers.TrainingArguments(
		output_dir=output_dir,
		do_train=True if not args.test else False,
		do_eval=True if not args.test and args.datalist_cv!="" else False,
		do_predict=True if args.test else False,
		num_train_epochs=args.nepochs,
		optim="adamw_torch",
		lr_scheduler_type=args.lr_scheduler,
		learning_rate=args.lr,
		warmup_ratio=args.warmup_ratio,
		#warmup_steps=num_warmup_steps,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=batch_size_eval,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		dataloader_drop_last= args.drop_last,
		eval_strategy=eval_strategy,
		eval_on_start=args.run_eval_on_start,
		eval_steps=args.logging_steps,
		metric_for_best_model=args.metric_for_best_model,
		greater_is_better=True,
		load_best_model_at_end=load_best_model_at_end,
		##batch_eval_metrics=False,
		##label_names=label_names,# DO NOT USE (see https://discuss.huggingface.co/t/why-do-i-get-no-validation-loss-and-why-are-metrics-not-calculated/32373)
		#save_strategy="epoch" if args.save_model_every_epoch else "no",
		save_strategy=save_strategy,
		save_total_limit=args.max_checkpoints, # at most keep only BEST + LAST
		logging_dir = log_dir,
		log_level="debug",
		logging_strategy="steps",
		logging_first_step=True,
		logging_steps=args.logging_steps,
		logging_nan_inf_filter=False,
		#disable_tqdm=True,
		run_name=args.runname,
		#report_to="wandb",  # enable logging to W&B
		report_to=args.report_to,
		seed=args.seed,
		dataloader_num_workers=args.num_workers,
		dataloader_pin_memory=(args.pin_memory=="true"),
		dataloader_persistent_workers=(args.persistent_workers=="true" and args.num_workers>0),
		ddp_find_unused_parameters=args.ddp_find_unused_parameters,
		fp16=args.fp16,
		bf16=args.bf16,
		weight_decay=args.weight_decay,
		remove_unused_columns=False if args.data_modality=="multimodal" else True,
	)
	
	print("--> training options")
	print(training_opts)		
			
	return training_opts		

##########################
##  SAVE METRICS CURVE
##########################
def save_curves_csv_from_predictions(
	trainer, 
	dataset, 
	outfile_csv: str, 
	num_ticks: int = 1001
):
	"""
		Uses trainer.predict(dataset) to get logits & labels, builds curves via metrics.binary_curves_from_probs,
		and writes curves to CSV. Writes BSS as a one-line sidecar .bss.txt file.
	"""
	# Predict to get logits & labels
	pred_out = trainer.predict(dataset)
	logits = torch.as_tensor(pred_out.predictions)
	if logits.ndim == 1:
		logits = logits.view(-1, 1)

	# Build positive-class probabilities
	if logits.shape[1] == 1:
		p_pos = torch.sigmoid(logits).squeeze(1).cpu().numpy()
	else:
		p = torch.softmax(logits, dim=1)
		p_pos = p[:, 1].detach().cpu().numpy()

	y_true = np.asarray(pred_out.label_ids, dtype=np.int64)

	curves = binary_curves_from_probs(
		p_pos=p_pos,
		y_true=y_true,
		num_ticks=num_ticks,
	)

	thresholds = curves["thresholds"]
	precision  = curves["precision"]
	recall     = curves["recall"]
	f1         = curves["f1"]
	tss        = curves["tss"]
	hss        = curves["hss"]
	mcc        = curves["mcc"]
	apss       = curves["apss"]
	
	os.makedirs(os.path.dirname(outfile_csv), exist_ok=True)

	# Write CSV
	with open(outfile_csv, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["threshold", "precision", "recall", "f1", "tss", "hss", "mcc", "apss"])
		for i in range(len(thresholds)):
			w.writerow([
				float(thresholds[i]),
				float(precision[i]),
				float(recall[i]),
				float(f1[i]),
				float(tss[i]),
				float(hss[i]),
				float(mcc[i]),
				float(apss[i]),
			])

	# Save BSS alongside (single number)
	#with open(outfile_csv.replace(".csv", ".bss.txt"), "w") as f:
	#	f.write(f"BSS={bss:.6f}\n")

			
##################
##   RUN TEST   ##
##################
def run_test(
	trainer,
	dataset,
	args
):
	""" Run test """
	
	# - Run model test		
	predictions, labels, metrics= trainer.predict(dataset, metric_key_prefix="predict")
	
	# - Print & save metrics	
	print("--> predictions")
	print(type(predictions))
	print(predictions)
		
	print("--> labels")
	print(type(labels))
	print(labels)
		
	print("--> prediction metrics")
	print(metrics) 
		
	trainer.log_metrics("predict", metrics)
	trainer.save_metrics("predict", metrics)			

	# - Save metric curves			
	#if args.compute_metrics_vs_thr and args.save_metric_curves:
	#	out_csv = os.path.join(args.outdir, "metrics_curves.csv")
	#	logger.info(f"Saving metric curves to {out_csv} ...")	
	#	save_curves_csv_from_predictions(trainer, dataset, out_csv, num_ticks=1001)
    	
		
##############################
##     RUN PREDICT
##############################
def run_predict(
	model,
	dataset,
	args,
	id2label,
	image_processor=None,
	data_collator=None,
	device="cuda:0"
):
	""" Run model predict """
	
	#device_choice= args.device
	#device = torch.device(device_choice if torch.cuda.is_available() else "cpu")

	inference_results= {"data": []}
	nsamples= dataset.get_sample_size()
	
	for i in range(nsamples):
		if i%1000==0:
			logger.info("#%d/%d images processed ..." % (i+1, nsamples))
		
		# - Retrieve image info
		image_info= dataset.load_image_info(i)
		sname= "sample_" + str(i+1)
		if "sname" in image_info:
			sname= image_info["sname"]
			
		# - Load image/video 
		if args.data_modality=="image":
			input_tensor= load_img_for_inference(
				dataset=dataset, 
				idx=i, 
				processor=image_processor if args.use_model_processor else None, 
				do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
				do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
				do_rescale=image_processor.do_rescale if args.use_model_processor else False                  # set to True only if processor should rescale
			)
			if input_tensor is None:
				logger.warning("Skip None tensor at index %d ..." % (i))
				continue
				
		elif args.data_modality=="video":
			input_tensor= load_video_for_inference(
				dataset=dataset, 
				idx=i, 
				processor=image_processor if args.use_model_processor else None, 
				do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
				do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
				do_rescale=image_processor.do_rescale if args.use_model_processor else False                  # set to True only if processor should rescale
			)
			if input_tensor is None:
				logger.warning("Skip None tensor at index %d ..." % (i))
				continue
				
		elif args.data_modality=="ts":
			input_batch= load_ts_for_inference(
				dataset=dataset, 
				idx=i,
				#device=device,
        #use_only_first_variate=True if args.ts_patching_mode=="time_only" else False
			)
			if input_batch is None:
				logger.warning("Skip None input batch at index %d ..." % (i))
				continue
				
		elif args.data_modality=="multimodal":
			if data_collator is None:
				raise ValueError("run_predict multimodal requires data_collator (VideoUni2TSMultimodalCollator).")

			item = dataset[i]
			if item is None:
				logger.warning("Skip None item at index %d ..." % (i))
				continue

			input_batch = data_collator([item])
			if input_batch is None:
				logger.warning("Skip None collated batch at index %d ..." % (i))
				continue		
				
		else:
			raise ValueError(f"Data modality {args.data_modality} not supported!")
						
		###input_tensor= input_tensor.unsqueeze(0).to(device)
		#input_tensor= input_tensor.to(device)

		if args.data_modality in ["ts", "multimodal"]:
			input_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_batch.items()}
		else:
			input_tensor = input_tensor.to(device)
 
		# - Compute model outputs
		with torch.no_grad():
			if args.data_modality in ["ts", "multimodal"]:
				outputs = model(**input_batch)
			else:
				outputs = model(input_tensor)

			logits = outputs.logits
				
		# - Compute predicted labels & probs
		if args.multilabel:
			sigmoid = torch.nn.Sigmoid()
			probs = sigmoid(logits.squeeze().cpu()).numpy()
			predictions = np.zeros(probs.shape)
			sigmoid_thr = getattr(args, "binary_thr", 0.5)
			predictions[np.where(probs >= sigmoid_thr)] = 1 # turn predicted id's into actual label names	
			predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
			predicted_probs = [float(probs[idx]) for idx, label in enumerate(predictions) if label == 1.0]
					
			# - Fill prediction results in summary dict
			image_info["label_pred"]= list(predicted_labels)
			image_info["prob_pred"]= list([float(item) for item in predicted_probs])
				
			if args.verbose:
				print("== Image: %s ==" % (sname))
				print("--> logits")
				print(logits)
				print("--> probs")
				print(probs)					
				print("--> predicted class id enc")
				print(predictions)
				print("--> predicted labels")
				print(predicted_labels)
				print("--> predicted probs")
				print(predicted_probs)
			
		elif args.ordinal:
			# --- Probabilistic decode (recommended) ---
			probsK = coral_logits_to_class_probs(logits)         # (K,)
			probs_np = probsK.cpu().numpy()
			class_id = int(probsK.argmax().item())
			predicted_label = id2label[class_id]                 # id2label must be the 4-class mapping
			predicted_prob = float(probs_np[class_id])

			# --- Optional: threshold-based ordinal decode (keeps ordinal semantics) ---
			# thresholds = (0.5, 0.5, 0.5)  # example for K=4
			# class_id_ord = coral_decode_with_thresholds(logits, thresholds=thresholds)
			# predicted_label = id2label[class_id_ord]
			# predicted_prob  = float(probs_np[class_id_ord])  # still report the class prob

			# - Fill prediction results in summary dict
			image_info["label_pred"] = str(predicted_label)
			image_info["prob_pred"]  = float(predicted_prob)
			
			# - Add extra info
			image_info["probs_all"]  = probs_np.tolist()                  # K-class probabilities
			image_info["ge_scores"]  = torch.sigmoid(logits.squeeze()).cpu().tolist()  # [>=C, >=M, >=X]

		else:

			lt = logits.detach().squeeze().cpu()
			binary_thr = getattr(args, "binary_thr", 0.5)

			if lt.ndim == 0 or lt.shape[-1] == 1:
				# SINGLE-LOGIT BINARY: logits shape (1,) or scalar
				p_pos = torch.sigmoid(lt).item()
				p_neg = 1.0 - p_pos
				# Decision: by threshold when single-logit, not argmax
				class_id = 1 if p_pos >= binary_thr else 0
				probs = np.array([p_neg, p_pos], dtype=np.float32)
				predicted_prob = float(p_pos if class_id == 1 else p_neg)

			else:
				# MULTICLASS or 2-LOGIT BINARY HEAD
				probs = torch.softmax(lt, dim=-1).numpy()
				class_id = int(np.argmax(probs))
				predicted_prob = float(probs[class_id])

				#softmax = torch.nn.Softmax(dim=0)
				#probs = softmax(logits.squeeze().cpu()).numpy()
				#class_id= np.argmax(probs)
				#predicted_label = id2label[class_id]
				#predicted_prob= probs[class_id]

			predicted_label = id2label[class_id]

			# - Fill prediction results in summary dict
			image_info["label_pred"]= str(predicted_label)
			image_info["prob_pred"]= float(predicted_prob)

			if args.verbose:
				print("== Image: %s ==" % (sname))
				print("logits.squeeze().cpu()")
				print(logits.squeeze().cpu())
				print(logits.squeeze().cpu().shape)
				print("--> probs")
				print(probs)
				print("--> predicted class id")
				print(class_id)
				print("--> predicted label")
				print(predicted_label)
				print("--> predicted probs")
				print(predicted_prob)
			
		inference_results["data"].append(image_info)
			
	# - Save json file
	logger.info("Saving inference results with prediction info to file %s ..." % (args.outfile))
	with open(args.outfile, 'w') as fp:
		json.dump(inference_results, fp, indent=2)
			
		
############################
##   RUN TRAIN
############################
def run_train(
	trainer,
	args
):
	""" Run model train """		
			
	################
	##  DEBUG
	#################
	if args.run_eval_on_start_manual:
		print("compute_metrics is None? ->", trainer.compute_metrics is None)
		metrics = trainer.evaluate()  # triggers evaluation once
		print("eval metrics keys ->", list(metrics.keys()))
	#################		
			
	# - Run train	
	#train_result = trainer.train(resume_from_checkpoint=checkpoint)
	train_result = trainer.train()
	
	# - Ensure all ranks finished training & any internal saves
	is_main = bool(trainer.args.should_save)
	out_dir = Path(trainer.args.output_dir)
	final_done = out_dir / ".final_done"
	best_done  = out_dir / ".best_done"
	#barrier_if_distributed()
	
	# - Save final model explicitly (if save_strategy is set to no)
	if is_main and trainer.args.save_strategy == "no":
		logger.info("Saving trained model ...")	
		trainer.save_model() # HF guards internally; only rank 0 writes	
		#trainer.save_model(trainer.args.output_dir)
		
	# - Wait so other ranks don't race reading/checking directories
	#barrier_if_distributed()
	
	# - Only the main process should create/update links
	if is_main:
		# - Link/copy "final" to the last checkpoint (if any), else to current model dir
		last_ckpt = find_last_checkpoint(out_dir) 
		if last_ckpt is None:
			# No checkpoints (e.g., save_strategy="no"): point "final" to current model dir
			logger.warning("⚠️ No checkpoints (e.g., save_strategy='no'): point 'final' to current model dir ...")
			safe_link_or_copy(out_dir, out_dir / "final")
		else:
			safe_link_or_copy(last_ckpt, out_dir / "final")
		touch(final_done)

		# - Best checkpoint link (only if validation happened and path exists)
		best_ckpt_path = getattr(trainer.state, "best_model_checkpoint", None)
		try:
			did_validation = (args.datalist_cv != "")
		except NameError:
			# If 'args' is not in scope, fall back to Trainer flags
			did_validation = bool(getattr(trainer.args, "load_best_model_at_end", False))
		
		if did_validation and best_ckpt_path:
			best_ckpt = Path(best_ckpt_path)
			if best_ckpt.exists():
				safe_link_or_copy(best_ckpt, out_dir / "best")
				touch(best_done)
			else:
				logger.warning(f"⚠️ Best checkpoint reported but not found on disk: {best_ckpt}")
				
		else:
			logger.info("ℹ️ No validation detected or no best checkpoint available; skipping 'best' link.")
		
	# - Final barrier so all ranks see consistent fs state before program exit
	#barrier_if_distributed()
	if not is_main:
		ok = wait_for_file(final_done, timeout_s=180)
		if not ok:
			logger.warning("Final model file save sentinel not seen, proceeding without blocking ...")
	
	# - Save metrics
	logger.info("Saving train metrics ...")        
	trainer.log_metrics("train", train_result.metrics)
	trainer.save_metrics("train", train_result.metrics)
	trainer.save_state()
		
	print("--> train metrics")
	print(train_result.metrics) 
        
	# - Run evaluation
	if args.datalist_cv!="":
		logger.info("Running model evaluation ...")
		metrics = trainer.evaluate()
		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", metrics)
		
		print("--> eval metrics")
		print(metrics) 		
				
##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	logger.info("Parse and retrieve input script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)", str(ex))
		return 1

	# - Read args
	datalist= args.datalist

	# - Run options
	#device_choice= args.device
	#device = torch.device(device_choice if torch.cuda.is_available() else "cpu")
	
	local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", -1))
	if local_rank != -1:
		device_choice = f"cuda:{local_rank}"
	else:
		device_choice = args.device
	device = torch.device(device_choice if torch.cuda.is_available() else "cpu")

	# - Model options
	modelname= args.model
	multiout= args.multiout
	
	# - Set config options
	id2label, label2id, id2target= get_target_maps(binary=args.binary, flare_thr=args.flare_thr)
	num_labels= len(id2label)  # - If binary this is =2
	nclasses= num_labels
	label_names= list(label2id.keys())
	
	print("id2label")
	print(id2label)
	print("label2id")
	print(label2id)
	print("id2target")
	print(id2target)
	print("num_labels")
	print(num_labels)
	print("nclasses")
	print(nclasses)
	print("label_names")
	print(label_names)
	
	#===========================
	#==   LOAD MODEL
	#===========================
	# - Load model & processor
	inference_mode= False
	if args.test or args.predict:
		inference_mode= True
	model, image_processor= load_model(args, id2label, label2id, num_labels, nclasses, inference_mode)
	
	# - Move model to device
	model= model.to(device)
	
	print("*** MODEL ***")
	print(model)
	print("")
	
	# - Freeze backbone (if enabled)
	logger.info("Applying model freeze (if enabled) ...")
	model= freeze_model(model, args)
	#if args.freeze_backbone:
	#	logger.info("Freezing model base layers ...")
	#	model= freeze_model(model, args)
		
	# - Print model layers
	logger.info("Printing frozen model layers ...")
	print_model(model, args, only_frozen=True, only_trainable=False, max_lines=1000)
	
	logger.info("Printing trainable model layers ...")
	print_model(model, args, only_frozen=False, only_trainable=True, max_lines=1000)
		
	if args.print_all_model_layers:	
		logger.info("Print entire model info ...")
		print_all_model_params(model)
		
	#	try:
	#		logger.info("Print base model info ...")	
	#		for name, param in model.base_model.named_parameters():
	#			print(name, param.requires_grad)	
	#	except Exception as e:
	#		logger.warning(f"Cannot print model base parameters (err={str(e)}, trying alternative method ...")	
	#		for name, param in model.backbone.named_parameters():
	#			print(name, param.requires_grad)	
				
	#	logger.info("Print entire model info ...")
	#	for name, param in model.named_parameters():
	#		print(name, param.requires_grad)	
	
	##################################
	##     DATASET
	##################################
	# - Create datasets
	logger.info("Creating datasets ...")
	dataset, dataset_cv= load_dataset(
		args, 
		image_processor,
		nclasses,
		id2target
	)
	
	# - Create data collators
	if args.data_modality=="image":
		data_collator= ImgDataCollator(
			image_processor=image_processor if args.use_model_processor else None, 
			do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
			do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
			do_rescale=image_processor.do_rescale if args.use_model_processor else False                  # set to True only if processor should rescale
		)
	
	elif args.data_modality=="video":
		data_collator= VideoDataCollator(
			image_processor=image_processor if args.use_model_processor else None, 
			do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
			do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
			do_rescale=image_processor.do_rescale if args.use_model_processor else False                  # set to True only if processor should rescale
		)
	
	elif args.data_modality=="ts":
		#data_collator= TSDataCollator()
		
		# - Compute ts var stats
		dataset.compute_ts_var_stats()
		print("--> ts stats")
		print(dataset.data_var_stats)
		
		if dataset_cv is not None:
			dataset_cv.compute_ts_var_stats()
			print("--> ts stats (EVAL)")
			print(dataset_cv.data_var_stats)
		
		# - Retrieve number of time series points
		ts_var_stats= dataset.data_var_stats
		ts_vars= list(ts_var_stats.keys())
		n_points= ts_var_stats[ts_vars[0]]["npoints"]
		if len(n_points)>1:
			logger.warning(f"Time series have more than one length: {str(n_points)} ...")
		context_length= n_points[0]

		data_collator = Uni2TSBatchCollator(
			context_length=context_length
		)
	
	elif args.data_modality=="multimodal":
	
		# - Compute ts var stats
		dataset.compute_ts_var_stats()
		print("--> ts stats")
		print(dataset._ts.data_var_stats)
		
		if dataset_cv is not None:
			dataset_cv.compute_ts_var_stats()
			print("--> ts stats (EVAL)")
			print(dataset_cv._ts.data_var_stats)
		
		# - Retrieve number of time series points
		ts_var_stats= dataset._ts.data_var_stats
		ts_vars= list(ts_var_stats.keys())
		n_points= ts_var_stats[ts_vars[0]]["npoints"]
		if len(n_points)>1:
			logger.warning(f"Time series have more than one length: {str(n_points)} ...")
		context_length= n_points[0]
		#context_length= int(args.ts_npoints)
		
		data_collator = VideoUni2TSMultimodalCollator(
			image_processor=image_processor if args.use_model_processor else None, 
			do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
			do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
			do_rescale=image_processor.do_rescale if args.use_model_processor else False,                  # set to True only if processor should rescale
			context_length=context_length,
			drop_none=True,
		)
	
	else:
		raise ValueError(f"Data modality {args.data_modality} not supported!")
				
	#######################################
	##     SET TRAINER
	#######################################
	# - Training options
	logger.info("Creating training options ...")
	training_opts= load_training_opts(args)
	
	# - Set metrics options
	chunk_size= training_opts.per_device_train_batch_size if dataset_cv is None else training_opts.per_device_eval_batch_size
	binary_thr= None if args.binary_thr==0.5 else args.binary_thr
	out_csv= None
	if args.save_metric_curves:
		out_csv = os.path.join(args.outdir, "metrics_curves.csv")
	
	# - Set metrics
	if args.multilabel:
		compute_metrics_custom= build_multi_label_metrics(label_names)
	elif args.ordinal:
		compute_metrics_custom= build_ordinal_metrics(label_names, thresholds=args.ordinal_thresholds)
	else:
		compute_metrics_custom= build_single_label_metrics(
			label_names, 
			chunk_size=chunk_size, 
			compute_best_tss=args.compute_best_tss, 
			compute_metrics_vs_thr=args.compute_metrics_vs_thr,
			binary_thr=binary_thr,
			curves_csv_path=out_csv
		)
		
	# - Compute class weights
	#num_labels = model.config.num_labels # this is modified in ordinal model
	class_weights= None
	class_weights_binary= None
	if args.use_weighted_loss:
		logger.info("Computing class weights from dataset ...")
		class_weights = dataset.compute_class_weights(
			num_classes=num_labels, 
			id2target=id2target,
			scheme=args.weight_compute_mode,
			normalize=args.normalize_weights,
			binary=False
		)
		print("--> CLASS WEIGHTS")
		print(class_weights)
		
		# - Compute binary class weights?
		if args.binary: 
			logger.info("Computing binary class weights from dataset ...")
			class_weights_binary= dataset.compute_class_weights(
				num_classes=num_labels, 
				id2target=id2target,
				scheme=args.weight_compute_mode,
				normalize=args.normalize_weights,
				binary=True,
				positive_label=1, 
				laplace=1.0
			)
			
			print("--> BINARY CLASS WEIGHTS")
			print(class_weights_binary)
		
	# - Compute ordinal pos weights
	ordinal_pos_weights= None
	if args.use_weighted_loss:
		logger.info("Computing ordinal pos_weight from dataset ...")
		ordinal_pos_weights, _= dataset.compute_ordinal_pos_weight(
			num_classes=num_labels, 
			id2target=id2target, 
			eps=1e-12, 
			clip_max=200, # clip if a class is missing otherwise pos weights becomes huge
			#device="cuda" if torch.cuda.is_available() else "cpu"
			device=device
		)
		print("--> ORDINAL POS WEIGHTS")
		print(ordinal_pos_weights)
	
	# - Compute weights for data sampler
	sample_weights = None
	if args.use_weighted_sampler:
		if args.sample_weight_from_flareid:
			logger.info("Computing sample weights from dataset flare_id data ...")
			sample_weights = dataset.compute_sample_weights_from_flareid(
				num_classes=4,
				scheme=args.sample_weight_compute_mode,
				normalize=args.normalize_weights
			)
		else:
			logger.info("Computing sample weights from dataset target data ...")
			sample_weights = dataset.compute_sample_weights(
				num_classes=num_labels,
				id2target=id2target,
				scheme=args.sample_weight_compute_mode,
				normalize=args.normalize_weights
			)
			
		print("--> SAMPLE WEIGHTS")
		print(f"min: {min(sample_weights)}, max: {max(sample_weights)}, mean: {np.mean(sample_weights)}")
		
			
	# Set focal loss pars
	#   - For focal alpha in multiclass, you can re-use class_weights
	#   - Often alpha ~ class_weights (normalized); you can also pass a float
	focal_alpha= None
	if args.loss_type=="focal":
		if args.set_focal_alpha_to_mild_estimate:
			logger.info("Setting focal alpha to mild estimate ...")
			focal_alpha, counts= dataset.compute_mild_focal_alpha_from_dataset(
    		num_classes=4 if args.sample_weight_from_flareid else num_labels,
    		id2target=id2target,
				exponent=0.5,
				cap_ratio=10.0,
				#device="cuda" if torch.cuda.is_available() else "cpu"
				device=device
			)
		else:
			logger.info("Setting focal alpha to class_weights if not None ...")
			counts= None
			focal_alpha = class_weights if args.loss_type == "focal" else None	

		def summarize_alpha(alpha_t, counts=None):
			if alpha_t is None:
				print("Focal alpha: None")
			else:
				alpha = alpha_t.detach().cpu().numpy()
				print("Focal alpha  :", np.round(alpha, 4).tolist(), " (mean=", round(alpha.mean(), 4), ", median=", round(np.median(alpha), 4), ")")
			
			if counts is not None:
				print("Class counts :", counts.tolist())
		
		print("--> FOCAL GAMMA")
		print(args.focal_gamma)
		print("--> FOCAL ALPHA")
		print(focal_alpha)
		summarize_alpha(focal_alpha, counts)
	
	# - Debug printout
	if hasattr(model, "config"):
		print("model.config.id2label:", model.config.id2label)
		print("model.config.label2id:", model.config.label2id)
		
	# - Set trainer
	#   NB: choose trainer class by modality
	if args.data_modality == "ts":
		logger.info("Using custom trainer for time-series data ...")
		TrainerClass = CustomTrainerTS
	elif args.data_modality == "video":
		if args.video_model=="imgfeatts":
			#TrainerClass = CustomTrainerTS
			TrainerClass = CustomTrainer
		else:
			TrainerClass = CustomTrainer
	else:
		logger.info("Using custom trainer ...")
		TrainerClass = CustomTrainer
	
	trainer = TrainerClass(
		model=model,
		args=training_opts,
		train_dataset=dataset,
		eval_dataset=dataset_cv,
		compute_metrics=compute_metrics_custom,
		processing_class=image_processor,
		data_collator=data_collator,
		class_weights=class_weights,
		multilabel=bool(args.multilabel),
		loss_type=args.loss_type,                  # "ce" or "focal"
		focal_gamma=args.focal_gamma,
		focal_alpha=focal_alpha,              # tensor[C] or float or None
		sample_weights=sample_weights,        # enables WeightedRandomSampler
		sol_score=args.sol_score,
		sol_distribution=args.sol_distribution,
		sol_mode=args.sol_mode,
		sol_add_constant=args.sol_add_constant,
		ordinal=bool(args.ordinal),
		ordinal_pos_weights=ordinal_pos_weights,
		compute_train_metrics=args.compute_train_metrics,
		binary_pos_weights=class_weights_binary,
		logitout_size=(1 if args.binary else num_labels),
		verbose=args.verbose
	)
		
	if args.compute_train_metrics:
		trainer.add_callback(TrainMetricsCallback(trainer))
		
	if args.clear_eval_cache:
		trainer.add_callback(CudaGCCallback(trainer))
		
	#######################################
	##     RUN TEST
	#######################################		
	# - Run predict on test set
	if args.test:
		logger.info("Predict model on input data %s ..." % (args.datalist))
		run_test(trainer, dataset, args)
	
	################################
	##    RUN PREDICT
	################################
	# - Run predict
	elif args.predict:
		logger.info("Running model inference on input data %s ..." % (args.datalist))
		run_predict(model, dataset, args, id2label, image_processor, data_collator=data_collator, device=device)
	
	################################
	##    TRAIN
	################################
	# - Run model train
	else:
		logger.info("Run model training ...")
		run_train(trainer, args)	
	
	
	return 0
		
###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
