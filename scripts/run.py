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

# - SKLEARN
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import hamming_loss

# - TORCH
import torch
from torch.optim import AdamW
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# - TRANSFORMERS
import transformers
from transformers import Trainer
from transformers import pipeline
from transformers import AutoProcessor, AutoModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import ViTForImageClassification, ViTConfig
from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
from transformers import VideoMAEForVideoClassification
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
from sfforecaster.model import MultiHorizonVideoMAE
from sfforecaster.utils import *
from sfforecaster.dataset import get_target_maps
from sfforecaster.dataset import VideoDataset, ImgDataset, ImgStackDataset
from sfforecaster.custom_transforms import FlippingTransform, Rotate90Transform
from sfforecaster.custom_transforms import VideoFlipping, VideoResize, VideoNormalize, VideoRotate90 
from sfforecaster.metrics import build_multi_label_metrics, build_single_label_metrics, build_ordinal_metrics
from sfforecaster.trainer import CustomTrainer, TrainMetricsCallback
from sfforecaster.trainer import VideoDataCollator, ImgDataCollator
from sfforecaster.model import CoralOrdinalHead
from sfforecaster.inference import coral_logits_to_class_probs, coral_decode_with_thresholds
from sfforecaster.inference import load_img_for_inference, load_video_for_inference

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

	# - Image pre-processing options
	parser.add_argument('--zscale', dest='zscale', action='store_true',help='Apply zscale transform to input images (default=false)')	
	parser.set_defaults(zscale=False)
	parser.add_argument('-zscale_contrast', '--zscale_contrast', dest='zscale_contrast', required=False, type=float, default=0.25, action='store', help='zscale contrast parameter (default=0.25)')
	parser.add_argument('--grayscale', dest='grayscale', action='store_true',help='Load input images in grayscale (1 chan tensor) (default=false)')	
	parser.set_defaults(grayscale=False)
	parser.add_argument('--resize', dest='resize', action='store_true', help='Resize input image before model processor. If false the model processor will resize anyway to its image size (default=false)')	
	parser.set_defaults(resize=False)
	parser.add_argument('-resize_size', '--resize_size', dest='resize_size', required=False, type=int, default=224, action='store', help='Resize size in pixels used if --resize option is enabled (default=224)')	
	
	# - Model options
	parser.add_argument('-model', '--model', dest='model', required=False, type=str, default="google/siglip-so400m-patch14-384", action='store', help='Model pretrained file name or weight path to be loaded {google/siglip-large-patch16-256, google/siglip-base-patch16-256, google/siglip-base-patch16-256-i18n, google/siglip-so400m-patch14-384, google/siglip-base-patch16-224, MCG-NJU/videomae-base, MCG-NJU/videomae-large, OpenGVLab/VideoMAEv2-Large}')
	parser.add_argument('--videoloader', dest='videoloader', action='store_true',help='Use video loader (default=false)')	
	parser.set_defaults(videoloader=False)
	parser.add_argument('--vitloader', dest='vitloader', action='store_true', help='If enabled use ViTForImageClassification to load model otherwise AutoModelForImageClassification (default=false)')	
	parser.set_defaults(vitloader=False)
	
	parser.add_argument('--use_model_processor', dest='use_model_processor', action='store_true', help='Use model image processor in data collator (default=false)')	
	parser.set_defaults(use_model_processor=False)
	
	parser.add_argument('--binary', dest='binary', action='store_true',help='Choose binary classification label scheme (default=false)')	
	parser.set_defaults(binary=False)
	parser.add_argument('-flare_thr', '--flare_thr', dest='flare_thr', required=False, type=str, default='C', action='store',help='Choose flare class label name: {C-->label=C+,M-->label=M+}.')
	
	parser.add_argument('--multilabel', dest='multilabel', action='store_true',help='Do multilabel classification (default=false)')	
	parser.set_defaults(multilabel=False)
	parser.add_argument('--multiout', dest='multiout', action='store_true',help='Do multi-step forecasting classification (default=false)')	
	parser.set_defaults(multiout=False)
	parser.add_argument('-num_horizons', '--num_horizons', dest='num_horizons', required=False, type=int, default=3, action='store',help='Number of forecasting horizons (default=3)')
	
	parser.add_argument('--ordinal', dest='ordinal', action='store_true',help='Load ordinal head model for classification (default=false)')	
	parser.set_defaults(ordinal=False)
	
	#parser.add_argument('--skip_first_class', dest='skip_first_class', action='store_true',help='Skip first class (e.g. NONE/BACKGROUND) in multilabel classifier (default=false)')	
	#parser.set_defaults(skip_first_class=False)
	
	parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true',help='Make backbone layers are non-tranable (default=false)')	
	parser.set_defaults(freeze_backbone=False)
	parser.add_argument('-max_freeze_layer_id', '--max_freeze_layer_id', dest='max_freeze_layer_id', required=False, type=int, default=-1, action='store',help='ID of the last layer kept frozen. -1 means all are frozen if --freeze_backbone option is enabled (default=-1)')
	
	# - Model training options
	parser.add_argument('--run_eval_on_start', dest='run_eval_on_start', action='store_true',help='Run model evaluation on start for debug (default=false)')	
	parser.set_defaults(run_eval_on_start=False)
	parser.add_argument('-logging_steps', '--logging_steps', dest='logging_steps', required=False, type=int, default=1, action='store',help='NUmber of logging steps (default=1)')
	parser.add_argument('--run_eval_on_step', dest='run_eval_on_step', action='store_true',help='Run model evaluation after each step (default=false)')	
	parser.set_defaults(run_eval_on_step=False)
	parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', dest='gradient_accumulation_steps', required=False, type=int, default=1, action='store',help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass (default=1)')

	parser.add_argument('-ngpu', '--ngpu', dest='ngpu', required=False, type=int, default=1, action='store',help='Number of gpus used for the run. Needed to compute the global number of training steps (default=1)')	
	parser.add_argument('-nepochs', '--nepochs', dest='nepochs', required=False, type=int, default=1, action='store',help='Number of epochs used in network training (default=100)')	
	#parser.add_argument('-optimizer', '--optimizer', dest='optimizer', required=False, type=str, default='adamw', action='store',help='Optimizer used (default=rmsprop)')
	parser.add_argument('-lr_scheduler', '--lr_scheduler', dest='lr_scheduler', required=False, type=str, default='constant', action='store',help='Learning rate scheduler used {constant, linear, cosine, cosine_with_min_lr} (default=cosine)')
	parser.add_argument('-lr', '--lr', dest='lr', required=False, type=float, default=5e-5, action='store',help='Learning rate (default=5e-5)')
	#parser.add_argument('-min_lr', '--min_lr', dest='min_lr', required=False, type=float, default=1e-6, action='store',help='Learning rate min used in cosine_with_min_lr (default=1.e-6)')
	parser.add_argument('-warmup_ratio', '--warmup_ratio', dest='warmup_ratio', required=False, type=float, default=0.2, action='store',help='Warmup ratio par (default=0.2)')
	parser.add_argument('-batch_size', '--batch_size', dest='batch_size', required=False, type=int, default=8, action='store',help='Batch size used in training (default=8)')
	
	parser.add_argument('--drop_last', dest='drop_last', action='store_true',help='Drop last incomplete batch (default=false)')	
	parser.set_defaults(drop_last=False)
	
	# - Imbalanced trainer options
	parser.add_argument("--use_custom_trainer", dest='use_custom_trainer', action="store_true", default=False, help="Use custom trainer (for imbalance).")
	parser.add_argument("--use_weighted_loss", dest='use_weighted_loss', action="store_true", default=False, help="Use class-weighted loss (CE or focal alpha).")
	parser.add_argument("--use_weighted_sampler", dest='use_weighted_sampler', action="store_true", default=False, help="Use a WeightedRandomSampler for training.")
	parser.add_argument("--sample_weight_from_flareid", dest='sample_weight_from_flareid', action="store_true", default=False, help="Compute sample weights from flare id (mostly used for binary class).")
	parser.add_argument("--weight_compute_mode", dest='weight_compute_mode', type=str, choices=["balanced", "inverse", "inverse_v2"], default="balanced", help="How to compute class/sample weights")
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
		
	# - Run options
	parser.add_argument('-device', '--device', dest='device', required=False, type=str, default="cuda:0", action='store',help='Device identifier')
	parser.add_argument('-runname', '--runname', dest='runname', required=False, type=str, default="llava_1.5_radio", action='store',help='Run name')
	parser.add_argument('--verbose', dest='verbose', action='store_true',help='Enable verbose printout (default=false)')	
	parser.set_defaults(verbose=False)
	
	parser.add_argument('--predict', dest='predict', action='store_true', help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)
	parser.add_argument('--test', dest='test', action='store_true', help='Run model test on input data (default=false)')	
	parser.set_defaults(test=False)
	
	parser.add_argument("--report_to", dest='report_to', type=str, default="wandb", help="Report logs/metrics to {wandb, none}")

	# - Output options
	parser.add_argument('-outdir','--outdir', dest='outdir', required=False, default="", type=str, help='Output data dir') 
	parser.add_argument('--save_model_every_epoch', dest='save_model_every_epoch', action='store_true', help='Save model every epoch (default=false)')	
	parser.set_defaults(save_model_every_epoch=False)
	parser.add_argument('-max_checkpoints', '--max_checkpoints', dest='max_checkpoints', required=False, type=int, default=1, action='store',help='Max number of saved checkpoints (default=1)')
	parser.add_argument('-outfile','--outfile', dest='outfile', required=False, default="classifier_results.json", type=str, help='Output file with saved inference results') 
	
	args = parser.parse_args()	

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
	

def load_image_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses
):
	""" Load image model & processor """

	#===================================
	#==     MULTI-OUT MODEL
	#===================================
	# - Load model
	if args.vitloader:
		config= ViTConfig.from_pretrained(
			args.model,
			problem_type="single_label_classification", 
			id2label=id2label, 
			label2id=label2id,
			num_labels=num_labels
		)
		
		if multiout:
			model = MultiHorizonViT.from_pretrained(
				args.model,
				config=config,
				num_horizons=args.num_horizons,
				num_classes=nclasses
			)

		else:
			model= ViTForImageClassification.from_pretrained(
				args.model,
				config=config
			)
				
		# - Load processor	
		image_processor = ViTImageProcessor.from_pretrained(args.model)
	
	#===================================
	#==     SINGLE-OUT MODEL
	#===================================
	else:
	
		if args.ordinal:
			# - Load ordinal-head model
			model= load_ordinal_image_model(args, nclasses)
		else:
			# - Load standard model
			model = AutoModelForImageClassification.from_pretrained(
				args.model, 
				problem_type="single_label_classification", 
				id2label=id2label, 
				label2id=label2id,
				num_labels=num_labels
			)
		
		# - Load processor	
		image_processor = AutoImageProcessor.from_pretrained(args.model)
		
	return model, image_processor	
	
						
			
def load_video_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses
):
	""" Load video model & processor """
	
	#===================================
	#==     MULTI-OUT MODEL
	#===================================
	if args.multiout:
		# - Load config
		config = VideoMAEConfig.from_pretrained(
			args.model,
			problem_type="single_label_classification", 
			id2label=id2label, 
			label2id=label2id,
			num_labels=num_labels
		)
		
		# - Load model
		model = MultiHorizonVideoMAE.from_pretrained(
			args.model,
			config=config,
			num_horizons=args.num_horizons,
			num_classes=nclasses
		)

	#===================================
	#==     SINGLE-OUT MODEL
	#===================================
	else:
		# - Load model
		model = VideoMAEForVideoClassification.from_pretrained(
			args.model,
			problem_type="single_label_classification", 
			id2label=id2label, 
			label2id=label2id,
			num_labels=num_labels
			#attn_implementation="sdpa", # "flash_attention_2" 
			#torch_dtype=torch.float16,  # "auto"
		)
	
	# - Load processor
	image_processor = VideoMAEImageProcessor.from_pretrained(args.model)
	
	return model, image_processor
		
			
def load_model(
	args,
	id2label,
	label2id,
	num_labels,
	nclasses
):
	""" Load model & processor """
	
	if args.videoloader:
		return load_video_model(args, id2label, label2id, num_labels, nclasses)
	else:
		return load_image_model(args, id2label, label2id, num_labels, nclasses)


def freeze_model(model, args):
	""" Freeze certain part of the model """
	
	logger.info("Freezing model base layers ...")
	for name, param in model.base_model.named_parameters():	
		if name.startswith("vision_model.encoder"):
			layer_index= extract_layer_id(name)
			if args.max_freeze_layer_id==-1 or (args.max_freeze_layer_id>=0 and layer_index!=-1 and layer_index<args.max_freeze_layer_id):
				param.requires_grad = False
		
	return model


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

	if args.videoloader:
		transform_train= T.Compose([
			VideoResize(size, interpolation=T.InterpolationMode.BICUBIC),
			VideoFlipping(),
			VideoRotate90(),
			VideoNormalize(mean=mean, std=std),
		])
		
		transform= T.Compose([
			VideoResize(size, interpolation=T.InterpolationMode.BICUBIC),
			VideoNormalize(mean=mean, std=std),
		])
	
	else:
		transform_train = T.Compose(
			[
				T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
				FlippingTransform(),
				Rotate90Transform(),
				#T.ToTensor(),
				T.Normalize(mean=mean, std=std),
			]
		)
	
		transform = T.Compose(
			[
				T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
				#T.ToTensor(),
				T.Normalize(mean=mean, std=std),
			]
		)
	
	return transform_train, transform
	

def load_transform(args, image_processor):
	""" Load data transform """
		
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
	blur_aug= T.GaussianBlur(kernel_size, sigma=(sigma_min, sigma_max))

	transform_train = T.Compose(
		[
			T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
			FlippingTransform(),
			Rotate90Transform(),
			#T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		]
	)
	
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
	#==   CREATE DATA TRANSFORMS
	#====================================
	# - Load data transforms
	if args.videoloader:
		transform_train, transform_valtest= load_video_transform(args, image_processor)
	else:
		transform_train, transform_valtest= load_transform(args, image_processor)
	
	#====================================
	#==   CREATE DATASET
	#====================================
	# - Init stuff
	dataset_cv= None
	dataset= None
	nsamples= 0
	nsamples_cv= 0
	DatasetClass= None
	if args.videoloader:
		DatasetClass= VideoDataset
	else:
		DatasetClass= ImgDataset
	
	transform= transform_train
	if args.predict or args.test:
		transform= transform_valtest
		
	# - Create train (or test set if args.predict or args.test)
	if args.predict or args.test:
		logger.info("Create dataset for prediction/test ...")
	else:
		logger.info("Create train dataset ...")
	
	dataset= DatasetClass(
		filename=args.datalist,
		transform=transform,
		load_as_gray=args.grayscale,
		apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
		resize=args.resize, resize_size=args.resize_size,
		nclasses=nclasses,
		id2target=id2target,
		multiout=args.multiout,
		multilabel=args.multilabel,
		ordinal=args.ordinal,
		verbose=args.verbose
	)
		
	nsamples= dataset.get_sample_size()
	
	logger.info("#%d entries in dataset ..." % (nsamples))
		
	# - Create validation set?
	if args.datalist_cv!="":
		dataset_cv= DatasetClass(
			filename=args.datalist_cv,
			transform=transform_valtest,
			load_as_gray=args.grayscale,
			apply_zscale=args.zscale, zscale_contrast=args.zscale_contrast,
			resize=args.resize, resize_size=args.resize_size,
			nclasses=nclasses,
			id2target=id2target,
			multiout=args.multiout,
			multilabel=args.multilabel,
			ordinal=args.ordinal,
			verbose=args.verbose
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
	
	# - Set eval strategy
	eval_strategy= "no"
	if args.datalist_cv!="":
		if args.run_eval_on_step:
			eval_strategy= "steps"
		else:
			eval_strategy= "epoch"
	
	# - Set training options
	logger.info("Set model options ...")
	training_opts= transformers.TrainingArguments(
		output_dir=output_dir,
		do_train=True if not args.test else False,
		do_eval=True if not args.test and args.datalist_cv!="" else False,
		do_predict=True if run_test else False,
		num_train_epochs=args.nepochs,
		optim="adamw_torch",
		lr_scheduler_type=args.lr_scheduler,
		learning_rate=args.lr,
		warmup_ratio=args.warmup_ratio,
		#warmup_steps=num_warmup_steps,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		dataloader_drop_last= args.drop_last,
		eval_strategy=eval_strategy,
		eval_on_start=args.run_eval_on_start,
		eval_steps=args.logging_steps,
		##batch_eval_metrics=False,
		##label_names=label_names,# DO NOT USE (see https://discuss.huggingface.co/t/why-do-i-get-no-validation-loss-and-why-are-metrics-not-calculated/32373)
		save_strategy="epoch" if args.save_model_every_epoch else "no",
		save_total_limit=args.max_checkpoints,
		logging_dir = log_dir,
		log_level="debug",
		logging_strategy="steps",
		logging_first_step=True,
		logging_steps=args.logging_steps,
		logging_nan_inf_filter=False,
		#disable_tqdm=True,
		run_name=args.runname,
    #report_to="wandb",  # enable logging to W&B
    report_to=args.report_to
	)
	
	print("--> training options")
	print(training_opts)		
			
	return training_opts		
			
##################
##   RUN TEST   ##
##################
def run_test(
	trainer,
	dataset
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
			
		
##############################
##     RUN PREDICT
##############################
def run_predict(
	model,
	dataset,
	args,
	id2label,
	image_processor=None
):
	""" Run model predict """
	
	device_choice= args.device
	device = torch.device(device_choice if torch.cuda.is_available() else "cpu")

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
		#if args.videoloader:
		#	input_tensor= dataset.load_video(i)
		#else: 
		#	input_tensor= dataset.load_image(i)
		
		if args.videoloader:
			input_tensor= load_video_for_inference(
				dataset=dataset, 
				idx=i, 
				processor=image_processor if args.use_model_processor else None, 
				do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
				do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
				do_rescale=image_processor.do_rescale if args.use_model_processor else False                  # set to True only if processor should rescale
			)
		else:
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
		#input_tensor= input_tensor.unsqueeze(0).to(device)
		input_tensor= input_tensor.to(device)
 
 		# - Compute model outputs
		with torch.no_grad():
			outputs = model(input_tensor)
			logits = outputs.logits
				
  	# - Compute predicted labels & probs
		if args.multilabel:
			sigmoid = torch.nn.Sigmoid()
			probs = sigmoid(logits.squeeze().cpu()).numpy()
			predictions = np.zeros(probs.shape)
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
			softmax = torch.nn.Softmax(dim=0)
			probs = softmax(logits.squeeze().cpu()).numpy()
			class_id= np.argmax(probs)
			predicted_label = id2label[class_id]
			predicted_prob= probs[class_id]
				
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
			
	# - Run train	
	#train_result = trainer.train(resume_from_checkpoint=checkpoint)
	train_result = trainer.train()
	
	# - Save model
	logger.info("Saving trained model ...")	
	trainer.save_model()

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
	logger.info("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		logger.error("Failed to get and parse options (err=%s)",str(ex))
		return 1

	# - Read args
	datalist= args.datalist
	videoloader= args.videoloader

	# - Run options
	device_choice= args.device
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
	logger.info("Loading model & processor (name=%s) ..." % (modelname))
	model, image_processor= load_model(args, id2label, label2id, num_labels, nclasses)
	
	# - Move model to device
	model= model.to(device)
	
	print("*** MODEL ***")
	print(model)
	print("")
	
	# - Freeze backbone?
	if args.freeze_backbone:
		logger.info("Freezing model base layers ...")
		model= freeze_model(model, args)
		
		logger.info("Print base model info ...")	
		for name, param in model.base_model.named_parameters():
			print(name, param.requires_grad)	
				
		logger.info("Print entire model info ...")
		for name, param in model.named_parameters():
			print(name, param.requires_grad)	
	
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
	
	# - Create collator fcn
	#def collate_fn(batch):
	#	pixel_values= []
	#	labels= []
	#	for item in batch:
	#		if item[0] is None:
	#			continue
	#		pixel_values.append(item[0])
	#		labels.append(item[1])
			
	#	pixel_values= torch.stack(pixel_values)
	#	labels= torch.stack(labels)
	#	return {"pixel_values": pixel_values, "labels": labels}
		
	# - Create data collators
	if args.videoloader:
		data_collator= VideoDataCollator(
			image_processor=image_processor if args.use_model_processor else None, 
			do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
			do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
			do_rescale=image_processor.do_rescale if args.use_model_processor else False                  # set to True only if processor should rescale
		)
	
	else:
		data_collator= ImgDataCollator(
			image_processor=image_processor if args.use_model_processor else None, 
			do_resize=image_processor.do_resize if args.use_model_processor else False,                   # set to True only if processor should resize
			do_normalize=image_processor.do_normalize if args.use_model_processor else False,              # set to True only if processor should normalize
			do_rescale=image_processor.do_rescale if args.use_model_processor else False                  # set to True only if processor should rescale
		)
	
	#######################################
	##     SET TRAINER
	#######################################
	# - Training options
	logger.info("Creating training options ...")
	training_opts= load_training_opts(args)
	
	# - Set metrics
	if args.multilabel:
		compute_metrics_custom= build_multi_label_metrics(label_names)
	elif args.ordinal:
		compute_metrics_custom= build_ordinal_metrics(label_names, thresholds=args.ordinal_thresholds)
	else:
		compute_metrics_custom= build_single_label_metrics(label_names)
		
	# - Compute class weights
	#num_labels = model.config.num_labels # this is modified in ordinal model
	class_weights= None
	if args.use_weighted_loss:
		logger.info("Computing class weights from dataset ...")
		class_weights = dataset.compute_class_weights(
			num_classes=num_labels, 
			id2target=id2target,
			scheme=args.weight_compute_mode,
			normalize=args.normalize_weights
		)
		print("--> CLASS WEIGHTS")
		print(class_weights)
		
	# - Compute ordinal pos weights
	ordinal_pos_weights= None
	if args.use_weighted_loss:
		logger.info("Computing ordinal pos_weight from dataset ...")
		ordinal_pos_weights, _= dataset.compute_ordinal_pos_weight(
			num_classes=num_labels, 
			id2target=id2target, 
			eps=1e-12, 
			clip_max=200, # clip if a class is missing otherwise pos weights becomes huge
			device="cuda" if torch.cuda.is_available() else "cpu"
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
				scheme=args.weight_compute_mode,
				normalize=args.normalize_weights
			)
		else:
			logger.info("Computing sample weights from dataset target data ...")
			sample_weights = dataset.compute_sample_weights(
				num_classes=num_labels,
				id2target=id2target,
				scheme=args.weight_compute_mode,
				normalize=args.normalize_weights
			)
		
	# Set focal loss pars
	#   - For focal alpha in multiclass, you can re-use class_weights
	#   - Often alpha ~ class_weights (normalized); you can also pass a float.
	if args.set_focal_alpha_to_mild_estimate:
		logger.info("Setting focal alpha to mild estimate ...")
		focal_alpha, counts= dataset.compute_mild_focal_alpha_from_dataset(
    	num_classes=4 if args.sample_weight_from_flareid else num_labels,
    	id2target=id2target,
			exponent=0.5,
			cap_ratio=10.0,
			device="cuda" if torch.cuda.is_available() else "cpu"
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
		
	
	if args.loss_type=="focal":
		print("--> FOCAL GAMMA")
		print(args.focal_gamma)
		print("--> FOCAL ALPHA")
		print(focal_alpha)
		summarize_alpha(focal_alpha, counts)
	
	# - Debug printout	
	print("id2label:", model.config.id2label)
	print("label2id:", model.config.label2id)
		
	# - Set trainer
	if args.use_custom_trainer:
		logger.info("Using custom class-weighted loss trainer ...")
		trainer = CustomTrainer(
			model=model,
			args=training_opts,
			train_dataset=dataset,
			eval_dataset=dataset_cv,
			compute_metrics=compute_metrics_custom,
			processing_class=image_processor,
			#data_collator=collate_fn,
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
			verbose=args.verbose
		)
		
		if args.compute_train_metrics:
			trainer.add_callback(TrainMetricsCallback(trainer))
		
	else:
		logger.info("Using standard trainer ...")
		trainer = Trainer(
			model=model,
			args=training_opts,
			train_dataset=dataset,
			eval_dataset=dataset_cv,
			compute_metrics=compute_metrics_custom,		
			processing_class=image_processor,
			#data_collator=collate_fn,
			data_collator=data_collator,
		)
	
	#######################################
	##     RUN TEST
	#######################################		
	# - Run predict on test set
	if args.test:
		logger.info("Predict model on input data %s ..." % (args.datalist))
		run_test(trainer, dataset)
	
	################################
	##    RUN PREDICT
	################################
	# - Run predict
	elif args.predict:
		logger.info("Running model inference on input data %s ..." % (args.datalist))
		run_predict(model, dataset, args, id2label, image_processor)
	
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
