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
from sfforecaster.dataset import VideoDataset, VideoMultiOutDataset
from sfforecaster.dataset import ImgDataset, ImgMultiOutDataset
from sfforecaster.dataset import ImgStackDataset, ImgStackMultiOutDataset
from sfforecaster.custom_transforms import FlippingTransform, Rotate90Transform
from sfforecaster.metrics import build_multi_label_metrics, build_single_label_metrics
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
	parser.add_argument('-model', '--model', dest='model', required=False, type=str, default="google/siglip-so400m-patch14-384", action='store', help='Model pretrained file name or weight path to be loaded {google/siglip-large-patch16-256, google/siglip-base-patch16-256, google/siglip-base-patch16-256-i18n, google/siglip-so400m-patch14-384, google/siglip-base-patch16-224, MCG-NJU/videomae-base, MCG-NJU/videomae-large}')
	parser.add_argument('--videoloader', dest='videoloader', action='store_true',help='Use video loader (default=false)')	
	parser.set_defaults(videoloader=False)
	parser.add_argument('--vitloader', dest='vitloader', action='store_true', help='If enabled use ViTForImageClassification to load model otherwise AutoModelForImageClassification (default=false)')	
	parser.set_defaults(vitloader=False)
	
	parser.add_argument('--multilabel', dest='multilabel', action='store_true',help='Do multilabel classification (default=false)')	
	parser.set_defaults(multilabel=False)
	parser.add_argument('--multiout', dest='multiout', action='store_true',help='Do multi-step forecasting classification (default=false)')	
	parser.set_defaults(multiout=False)
	parser.add_argument('-num_horizons', '--num_horizons', dest='num_horizons', required=False, type=int, default=3, action='store',help='Number of forecasting horizons (default=3)')
	
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

	# - Run options
	parser.add_argument('-device', '--device', dest='device', required=False, type=str, default="cuda:0", action='store',help='Device identifier')
	parser.add_argument('-runname', '--runname', dest='runname', required=False, type=str, default="llava_1.5_radio", action='store',help='Run name')
	parser.add_argument('--verbose', dest='verbose', action='store_true',help='Enable verbose printout (default=false)')	
	parser.set_defaults(verbose=False)
	
	parser.add_argument('--predict', dest='predict', action='store_true', help='Predict model on input data (default=false)')	
	parser.set_defaults(predict=False)
	parser.add_argument('--test', dest='test', action='store_true', help='Run model test on input data (default=false)')	
	parser.set_defaults(test=False)
	
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
		# - Load model
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
def load_transform(args, image_processor):
	""" Load data transform """
		
	# - Retrieve image processor transform parameters
	size = (image_processor.size["height"], image_processor.size["width"])
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
		eval_on_start=run_eval_on_start,
		eval_steps=logging_steps,
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
    report_to="wandb",  # enable logging to W&B
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
	id2label
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
		if args.videoloader:
			input_tensor= dataset.load_video(i)
		else: 
			input_tensor= dataset.load_image(i)
		
		if input_tensor is None:
			logger.warning("Skip None tensor at index %d ..." % (i))
			continue
		input_tensor= input_tensor.unsqueeze(0).to(device)
 
 		# - Compute model outputs
		with torch.no_grad():
			outputs = model(image_tensor)
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
					
		else:
			softmax = torch.nn.Softmax(dim=0)
			probs = softmax(logits.squeeze().cpu()).numpy()
			class_id= np.argmax(probs)
			predicted_label = id2label[class_id]
			predicted_prob= probs[class_id]
				
			# - Fill prediction results in summary dict
			image_info["label_pred"]= str(predicted_label)
			image_info["prob_pred"]= float(predicted_prob)
					
			if verbose:
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
	id2label, label2id, id2target= get_target_maps()
	num_labels= len(id2label)  # - If skip_first_class, this is =3
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
	def collate_fn(batch):
		pixel_values= []
		labels= []
		for item in batch:
			if item[0] is None:
				continue
			pixel_values.append(item[0])
			labels.append(item[1])
			
		pixel_values= torch.stack(pixel_values)
		labels= torch.stack(labels)
		return {"pixel_values": pixel_values, "labels": labels}
	
	#######################################
	##     SET TRAINER
	#######################################
	# - Training options
	logger.info("Creating training options ...")
	training_opts= load_training_opts(args)
	
	# - Set metrics
	if args.multilabel:
		compute_metrics_custom= build_multi_label_metrics(label_names)
	else:
		compute_metrics_custom= build_single_label_metrics(label_names)
		
	# - Set trainer
	trainer = Trainer(
		model=model,
		args=training_opts,
		train_dataset=dataset,
		eval_dataset=dataset_cv,
		compute_metrics=compute_metrics_custom,		
		processing_class=image_processor,
		data_collator=collate_fn
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
		run_predict(model, dataset, args, id2label)
	
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
