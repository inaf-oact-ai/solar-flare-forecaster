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
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# - TRANSFORMERS
from transformers.trainer_utils import EvalPrediction
from transformers import EvalPrediction    

# - SCLASSIFIER-VIT
from sfforecaster.utils import *
from sfforecaster import logger

##########################################
##    CUSTOM METRICS
##########################################
# - See https://stackoverflow.com/questions/32239577/getting-the-accuracy-for-multi-label-prediction-in-scikit-learn
def hamming_score(y_true, y_pred):
	""" Compute the hamming score """
	return ( (y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1) ).mean()
	
def hamming_score_v2(y_true, y_pred, normalize=True, sample_weight=None):
	""" Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case (http://stackoverflow.com/q/32239577/395857)"""
	acc_list = []
	for i in range(y_true.shape[0]):
		set_true = set( np.where(y_true[i])[0] )
		set_pred = set( np.where(y_pred[i])[0] )
		tmp_a = None
		if len(set_true) == 0 and len(set_pred) == 0:
			tmp_a = 1
		else:
			num= len(set_true.intersection(set_pred))
			denom= float( len(set_true.union(set_pred)) )
			tmp_a = num/denom
			
		acc_list.append(tmp_a)
		   
	return np.nanmean(acc_list)

###########################################
##   MULTI-LABEL CLASS METRICS
###########################################
def multi_label_metrics(predictions, labels, threshold=0.5, target_names=None):
	""" Helper function to compute multilabel metrics """
	
	# - First, apply sigmoid on predictions which are of shape (batch_size, num_labels)
	sigmoid = torch.nn.Sigmoid()
	probs = sigmoid(torch.Tensor(predictions))
	
	# - Next, use threshold to turn them into integer predictions
	y_pred = np.zeros(probs.shape)
	y_pred[np.where(probs >= threshold)] = 1
    
	# - Finally, compute metrics
	y_true = labels
	
	class_report= classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
	#class_report_str= classification_report(y_true, y_pred, target_names=target_names, output_dict=False)
	
	#accuracy = accuracy_score(y_true, y_pred)
	accuracy = accuracy_score(y_true, y_pred, normalize=True) # NB: This computes subset accuracy (the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true)
	
	precision= precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
	#precision= report['weighted avg']['precision']
	
	recall= recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
	#recall= report['weighted avg']['recall']    
	
	f1score= f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
	#f1score= report['weighted avg']['f1-score']
	
	f1score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
	roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
	h_loss= hamming_loss(y_true, y_pred)
	h_score= hamming_score_v2(y_true, y_pred)
	class_accuracy= [accuracy_score(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1]) ]  ## LIKELY NOT CORRECT!!!
	
	class_names= ''
	if target_names is not None:
		class_names= target_names
	else:
		print("y_true.shape")
		print(y_true.shape)
		nclasses= y_true.shape[-1]
		print("nclasses")
		print(nclasses)
		class_names= [str(item) for item in list(np.arange(0,nclasses))]
			
	print("class_names")
	print(class_names)  
    
	# - Return as dictionary
	metrics = {
		'class_names': class_names,
		'accuracy': accuracy,
		'recall': recall,
		'precision': precision,
		'f1score': f1score,
		'f1score_micro': f1score_micro,
		'roc_auc': roc_auc,
		'h_loss': h_loss,
		'h_score': h_score,
		'accuracy_class': class_accuracy,
		'class_report': class_report,
	}
	
	print("metrics")
	print(metrics)
	  
	return metrics


def build_multi_label_metrics(target_names):

	def compute_multi_label_metrics(p: EvalPrediction):
		""" Compute metrics """
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		nonlocal target_names
		
		# - Compute all metrics
		metrics = multi_label_metrics(
			predictions=preds, 
			labels=p.label_ids,
			target_names=target_names
		)
		
		# - Trainer wants only the scalar metrics
		metrics_scalar= {
			"accuracy": metrics["accuracy"],
			"precision": metrics["precision"],
			"recall": metrics["recall"],
			"f1score": metrics["f1score"],
			"f1score_micro": metrics["f1score_micro"]
		}
		
		return metrics_scalar
		
	return compute_multi_label_metrics

###########################################
##   SINGLE-LABEL CLASS METRICS
###########################################
def single_label_metrics(predictions, labels, target_names=None):
	""" Helper function to compute single label metrics """
	
	# - First, apply sigmoid on predictions which are of shape (batch_size, num_labels)
	softmax= torch.nn.Softmax(dim=1) # see https://discuss.pytorch.org/t/implicit-dimension-choice-for-softmax-warning/12314/8
	probs= softmax(torch.Tensor(predictions))
	
	# - Next, use threshold to turn them into integer predictions
	#y_pred= np.argmax(probs, axis=1)
	y_pred = torch.argmax(probs, dim=1).numpy()
	  
	# - Finally, compute metrics
	#   Ensure labels are NumPy array
	#y_true = np.where(labels==1)[1]
	#y_true = labels # already class indices
	y_true = labels if isinstance(labels, np.ndarray) else labels.detach().cpu().numpy()

	# If target names are provided, infer number of classes from them
	if target_names is not None:
		num_classes = len(target_names)
		label_indices = list(range(num_classes))
	else:
		# Fallback: infer from y_true and y_pred
		label_indices = sorted(set(y_true).union(set(y_pred)))
        
	class_report= classification_report(
		y_true, 
		y_pred, 
		labels=label_indices,
		target_names=target_names if target_names else None,
		output_dict=True
	)
	
	accuracy = accuracy_score(y_true, y_pred, normalize=True) # NB: This computes subset accuracy (the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true)
	
	precision= precision_score(y_true=y_true, y_pred=y_pred, average='weighted', labels=label_indices, zero_division=0)
	
	recall= recall_score(y_true=y_true, y_pred=y_pred, average='weighted', labels=label_indices, zero_division=0)
	
	f1score= f1_score(y_true=y_true, y_pred=y_pred, average='weighted', labels=label_indices, zero_division=0)
	
	f1score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro', labels=label_indices, zero_division=0)
	
	cm= confusion_matrix(y_true, y_pred, labels=label_indices)
	cm_norm= confusion_matrix(y_true, y_pred, labels=label_indices, normalize="true")

	print("confusion matrix")
	print(cm)

	print("confusion matrix (norm)")
	print(cm_norm)
	
	class_names= ''
	if target_names is not None:
		class_names= target_names
	else:
		nclasses= y_true.shape[-1]
		class_names= [str(item) for item in list(np.arange(0,nclasses))]
			
	# - Return as dictionary
	metrics = {
		'class_names': class_names,
		'accuracy': accuracy,
		'recall': recall,
		'precision': precision,
		'f1score': f1score,
		'f1score_micro': f1score_micro,
		'class_report': class_report,
		'confusion_matrix': cm.tolist(),# to make it serialized in json save
		'confusion_matrix_norm': cm_norm.tolist(), # to make it serialized in json save
	}
	
	print("--> metrics")
	print(metrics)
	  
	return metrics

def build_single_label_metrics(target_names):

	def compute_single_label_metrics(p: EvalPrediction):
		""" Compute metrics """
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		nonlocal target_names
		
		# - Compute all metrics
		metrics = single_label_metrics(
			predictions=preds, 
			labels=p.label_ids,
			target_names=target_names
		)
		
		# - Trainer wants only the scalar metrics
		metrics_scalar= {
			"accuracy": metrics["accuracy"],
			"precision": metrics["precision"],
			"recall": metrics["recall"],
			"f1score": metrics["f1score"],
			"f1score_micro": metrics["f1score_micro"]
		}
		
		return metrics_scalar
		
	return compute_single_label_metrics

