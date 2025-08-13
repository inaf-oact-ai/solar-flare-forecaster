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
from sklearn.metrics import matthews_corrcoef

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
	
def summarize_metrics_per_class(scores, support):
	"""
		scores: per-class array (e.g., HSS per class)
		support: number of true samples per class (cm.sum(axis=1))
	"""
	eps = 1e-7
	macro = np.nanmean(scores)
	weighted = np.nansum(scores * (support / (support.sum() + eps)))
	return {"macro": macro, "weighted": weighted, "per_class": scores}


def compute_micro_metrics_from_confusion_matrix(cm, eps=1e-7):
	"""
		Compute micro/global skill scores by first summing TP,FP,FN,TN across classes,
		then applying the binary formulas once to those totals.
	"""
	cm = cm.astype(np.float64)
	TP_c = np.diag(cm)
	FP_c = cm.sum(axis=0) - TP_c
	FN_c = cm.sum(axis=1) - TP_c
	TN_c = cm.sum() - (TP_c + FP_c + FN_c)

	# - Global totals (micro aggregation)
	TP = TP_c.sum()
	FP = FP_c.sum()
	FN = FN_c.sum()
	TN = TN_c.sum()
	N  = TP + FP + FN + TN

	# - Rates/precisions
	TPR = TP / (TP + FN + eps)      # recall/sensitivity
	TNR = TN / (TN + FP + eps)      # specificity
	PPV = TP / (TP + FP + eps)      # precision
	NPV = TN / (TN + FN + eps)
	FPR = FP / (FP + TN + eps)
	FNR = FN / (TP + FN + eps)
	FDR = FP / (TP + FP + eps)

	# - Skill scores from global totals
	TSS = TPR + TNR - 1

	# - Heidke Skill Score (equitable / HSS2) from global totals	
	HSS = (2.0 * (TP*TN - FP*FN)) / (
		(TP+FN)*(FN+TN) + (TP+FP)*(FP+TN) + eps
	)

	# Gilbert Skill Score / ETS
	# Expected random hits:
	TP_rand = (TP + FN) * (TP + FP) / (N + eps)
	GSS = (TP - TP_rand) / ((TP + FP + FN) - TP_rand + eps)

	overall_acc = (TP + TN) / (N + eps)

	return {
		"TP": TP, 
		"FP": FP, 
		"FN": FN, 
		"TN": TN, 
		"N": N,
		"TPR": TPR, 
		"TNR": TNR, 
		"PPV": PPV, 
		"NPV": NPV,
		"FPR": FPR, 
		"FNR": FNR, 
		"FDR": FDR,
		"TSS": TSS, 
		"HSS": HSS, 
		"GSS": GSS,
		"overall_accuracy": overall_acc
	}


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
	
	f1score_weighted= f1_score(y_true=y_true, y_pred=y_pred, average='weighted', labels=label_indices, zero_division=0)
	
	f1score_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro', labels=label_indices, zero_division=0)
	
	f1score_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro', labels=label_indices, zero_division=0)
	
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
		#nclasses= y_true.shape[-1]
		#class_names= [str(item) for item in list(np.arange(0,nclasses))]
		class_names = [str(i) for i in label_indices]
		
	# - Compute true/false positive/negative
	support = cm.sum(axis=1)
	FP = cm.sum(axis=0) - np.diag(cm)  
	FN = cm.sum(axis=1) - np.diag(cm)
	TP = np.diag(cm)
	TN = cm.sum() - (FP + FN + TP)
	
	eps= 1.e-7
	
	# - Sensitivity, hit rate, recall, or true positive rate
	TPR = TP/(TP + FN + eps)
	
	# - Specificity or true negative rate
	TNR = TN/(TN + FP + eps) 
	
	# - Precision or positive predictive value
	PPV = TP/(TP+FP)

	# - Negative predictive value
	NPV = TN/(TN+FN)

	# - Fall out or false positive rate
	FPR = FP/(FP+TN)

	# - False negative rate
	FNR = FN/(TP+FN)

	# - False discovery rate
	FDR = FP/(TP+FP)
	
	# - Overall accuracy
	ACC = (TP+TN)/(TP+FP+FN+TN)
	
	# - Compute True Skill Statistic (TSS)
	#TSS= ((TP*TN)-(FP*FN))/((TP+FN)*(FP+TN))
	TSS= TPR + TNR - 1
			
	# - Compute Heidke Skill Score (HSS)
	#HSS= 2*((TP*TN)-(FP*FN))/( ((TP+FN)*(TN+FN)) + ((FP+TN)*(FP+TP)) )
	HSS= 2*(TP*TN-FP*FN)/((TP+FN)*(TN+FN)+(TP+FP)*(FP+TN))
           
	# - Compute Gilbert Skill Score (GSS)
	GSS= (TP- ( ((TP+FN)*(TP+FP))/(TP+FP+TN+FN) ) )/( (TP+FP+FN) - ( ((TP+FN)*(TP+FP))/(TP+FP+TN+FN) ) )
	
	# - Compute Matthew’s correlation coefficient (MCC) (CHECK!!!)
	#MCC= ((TP*TN)-(FP*FN))/np.sqrt( (TP+FP)*(TP+FN)*(TP+FP)*(TN+FN) ) # WRONG in https://arxiv.org/pdf/2408.05590v1 ?
	MCC= ((TP*TN)-(FP*FN))/np.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )
	MCC_coeff= matthews_corrcoef(y_true=y_true, y_pred=y_pred)
			
	# - Compute summary metrics
	hss_summary = summarize_metrics_per_class(HSS, support)
	tss_summary = summarize_metrics_per_class(TSS, support)
	gss_summary = summarize_metrics_per_class(GSS, support)
	tpr_summary = summarize_metrics_per_class(TPR, support)	
	tnr_summary = summarize_metrics_per_class(TNR, support)	
			
	print(f"FP={FP}, FN={FN}, TP={TP}, TN={TN}, ACC={ACC}, accuracy={accuracy}, TSS={TSS}, HSS={HSS}, GSS={GSS}, MCC={MCC}, MCC_coeff={MCC_coeff}")
	print(f"HSS: {hss_summary}")
	print(f"TSS: {tss_summary}")
	print(f"GSS: {gss_summary}")
	print(f"TPR: {tpr_summary}")
	print(f"TNR: {tnr_summary}")
	
	# - Compute global metrics (as done in other papers)
	metrics_micro= compute_micro_metrics_from_confusion_matrix(cm)
	
	# - Return as dictionary
	metrics = {
		'class_names': class_names,
		'support': support,
		'accuracy': accuracy,
		'recall': recall,
		'precision': precision,
		'f1score_weighted': f1score_weighted,
		'f1score_micro': f1score_micro,
		'f1score_macro': f1score_macro,
		'class_report': class_report,
		'confusion_matrix': cm.tolist(),# to make it serialized in json save
		'confusion_matrix_norm': cm_norm.tolist(), # to make it serialized in json save
		'fp': FP,
		'fn': FN,
		'tp': TP,
		'tn': TN,
		'tpr': TPR,
		'tnr': TNR,
		'ppv': PPV,
		'npv': NPV,
		'fpr': FPR,
		'fnr': FNR,
		'fdr': FDR,
		'tss': TSS,
		'hss': HSS,
		'gss': GSS,
		'mcc': MCC_coeff,
		'tpr_summary': tpr_summary,
		'tnr_summary': tnr_summary,
		'hss_summary': hss_summary,
		'tss_summary': tss_summary,
		'gss_summary': gss_summary,
		'fp_micro': metrics_micro['FP'],
		'fn_micro': metrics_micro['FN'],
		'tp_micro': metrics_micro['TP'],
		'tn_micro': metrics_micro['TN'],
		'tpr_micro': metrics_micro['TPR'],
		'tnr_micro': metrics_micro['TNR'],
		'ppv_micro': metrics_micro['PPV'],
		'npv_micro': metrics_micro['NPV'],
		'fpr_micro': metrics_micro['FPR'],
		'fnr_micro': metrics_micro['FNR'],
		'fdr_micro': metrics_micro['FDR'],
		'tss_micro': metrics_micro['TSS'],
		'hss_micro': metrics_micro['HSS'],
		'gss_micro': metrics_micro['GSS'],
		'accuracy_micro': metrics_micro['overall_accuracy']
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
			"f1score_weighted": metrics["f1score_weighted"],
			"f1score_micro": metrics["f1score_micro"],
			"f1score_macro": metrics["f1score_macro"],
			#"fp": metrics["fp"],
			#"fn": metrics["fn"],
			#"tp": metrics["tp"],
			#"tn": metrics["tn"],
			#"tpr": metrics["tpr"],
			#"tnr": metrics["tnr"],
			#"ppv": metrics["ppv"],
			#"npv": metrics["npv"],
			#"fpr": metrics["fpr"],
			#"fnr": metrics["fnr"],
			#"fdr": metrics["fdr"],
			#"tss": metrics["tss"],
			#"hss": metrics["hss"],
			#"gss": metrics["gss"],
			"mcc": metrics["mcc"],
			"tpr_macro": metrics["tpr_summary"]["macro"],
			"tpr_weighted": metrics["tpr_summary"]["weighted"],
			"tnr_macro": metrics["tnr_summary"]["macro"],
			"tnr_weighted": metrics["tnr_summary"]["weighted"],
			"hss_macro": metrics["hss_summary"]["macro"],
			"tss_macro": metrics["tss_summary"]["macro"],
			"gss_macro": metrics["gss_summary"]["macro"],
			"hss_weighted": metrics["hss_summary"]["weighted"],
			"tss_weighted": metrics["tss_summary"]["weighted"],
			"gss_weighted": metrics["gss_summary"]["weighted"],
			"tpr_micro": metrics["tpr_micro"],
			"tnr_micro": metrics["tnr_micro"],
			"hss_micro": metrics["hss_micro"],
			"tss_micro": metrics["tss_micro"],
			"gss_micro": metrics["gss_micro"],
		}
		
		return metrics_scalar
		
	return compute_single_label_metrics

