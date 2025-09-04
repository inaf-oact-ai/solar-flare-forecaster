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
from sklearn.metrics import average_precision_score, cohen_kappa_score

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


def compute_micro_metrics_from_confusion_matrix_v2(y_true, y_pred, class_names, eps=1e-7):
	"""
		Compute micro/global skill scores by first summing TP,FP,FN,TN across classes,
		then applying the binary formulas once to those totals.
	"""
	
	# - Compute multilabel confusion matrix
	MCM = multilabel_confusion_matrix(
		y_true,
		y_pred,
		sample_weight=None,
		labels=class_names,
		samplewise=False
	)
	
	# - Compute TP/TN/FP/FN for each class
	TN_c= np.array([int(MCM[i].ravel()[0]) for i in range(MCM.shape[0])])
	FP_c= np.array([int(MCM[i].ravel()[1]) for i in range(MCM.shape[0])])
	FN_c= np.array([int(MCM[i].ravel()[2]) for i in range(MCM.shape[0])])
	TP_c= np.array([int(MCM[i].ravel()[3]) for i in range(MCM.shape[0])])
	
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
	
def best_tss_from_probs(p_pos, y_true, num_ticks=101, eps=1e-7):
	""" For binary classification: compute TSS vs threshold and select best TSS"""
	# p_pos: numpy array of P(class=1) ; y_true: {0,1}
	thrs = np.linspace(0.0, 1.0, num_ticks)
	best = {"tss": -1.0, "thr": 0.5, "tpr": 0.0, "tnr": 0.0}
	for t in thrs:
		y_pred = (p_pos >= t).astype(int)
		TP = ((y_true==1) & (y_pred==1)).sum()
		TN = ((y_true==0) & (y_pred==0)).sum()
		FP = ((y_true==0) & (y_pred==1)).sum()
		FN = ((y_true==1) & (y_pred==0)).sum()
		tpr = TP / (TP + FN + eps)
		tnr = TN / (TN + FP + eps)
		tss = tpr + tnr - 1.0
		if tss > best["tss"]:
			best = {"tss": tss, "thr": float(t), "tpr": tpr, "tnr": tnr}

	return best
	
def coral_probs_from_logits(logits_t: torch.Tensor) -> torch.Tensor:
	"""
		Convert CORAL/cumulative logits (B, K-1) into K-class probabilities (B, K).
		For K=4 (NONE<C<M<X), logits are [>=C, >=M, >=X].
	"""
	p_ge = torch.sigmoid(logits_t)              # (B, K-1)
	if p_ge.shape[1] == 3:  # K=4 expected
		p0 = 1.0 - p_ge[:, 0]                   # P(NONE)
		p1 = p_ge[:, 0] - p_ge[:, 1]            # P(C)
		p2 = p_ge[:, 1] - p_ge[:, 2]            # P(M)
		p3 = p_ge[:, 2]                         # P(X)
		probs = torch.stack([p0, p1, p2, p3], dim=1)
	else:
		# Generic K: probs[0]=1-p_ge[0], probs[k]=p_ge[k-1]-p_ge[k], probs[K-1]=p_ge[K-2]
		B, Km1 = p_ge.shape
		K = Km1 + 1
		comps = []
		comps.append(1.0 - p_ge[:, 0])
		for k in range(1, Km1):
			comps.append(p_ge[:, k-1] - p_ge[:, k])
		comps.append(p_ge[:, Km1-1])
		probs = torch.stack(comps, dim=1)
    
	# Numeric safety
	probs = torch.clamp(probs, min=1e-12)
	probs = probs / probs.sum(dim=1, keepdim=True)

	return probs  # (B, K)


def coral_decode_classes_from_logits(logits_t: torch.Tensor, thresholds=None) -> np.ndarray:
	"""
		Class indices from cumulative logits by counting passed thresholds.
		Default threshold=0.0 on logits (i.e., sigmoid>=0.5).
	"""
	if thresholds is None:
		ge = (logits_t >= 0.0).int()            # (B, K-1)
	else:
		thr = torch.as_tensor(thresholds, dtype=logits_t.dtype, device=logits_t.device)
		ge = (torch.sigmoid(logits_t) >= thr.view(1, -1)).int()

	return ge.sum(dim=1).detach().cpu().numpy()  # (B,)

def monotonicity_violations(logits_t: torch.Tensor) -> dict:
	"""
		Check p_ge monotonicity: p_ge[:,0] >= p_ge[:,1] >= ... >= p_ge[:,-1].
		Returns violation rate and mean positive violation magnitude.
	"""
	p = torch.sigmoid(logits_t)  # (B, K-1)
	diffs = p[:, :-1] - p[:, 1:]  # should be >= 0
	viol = (diffs < 0)
	rate = viol.any(dim=1).float().mean().item()
	mag = (-torch.minimum(diffs, torch.zeros_like(diffs))).mean().item()
	return {"mvr": rate, "mvm": mag}  # rate, mean violation magnitude


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
	#   NB: Distinguish the case of binary vs multiclass
	support = cm.sum(axis=1)
	
	binary_class= len(class_names) == 2
	if binary_class:
		TN, FP, FN, TP= cm.ravel()         # these are scalars
	else:
		FP = cm.sum(axis=0) - np.diag(cm)  # there are arrays
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
			
	print(f"FP={FP}, FN={FN}, TP={TP}, TN={TN}, ACC={ACC}, accuracy={accuracy}, TSS={TSS}, HSS={HSS}, GSS={GSS}, MCC={MCC}, MCC_coeff={MCC_coeff}")
	
	# - Compute summary metrics
	if not binary_class:
		hss_summary = summarize_metrics_per_class(HSS, support)
		tss_summary = summarize_metrics_per_class(TSS, support)
		gss_summary = summarize_metrics_per_class(GSS, support)
		tpr_summary = summarize_metrics_per_class(TPR, support)	
		tnr_summary = summarize_metrics_per_class(TNR, support)	
			
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
		'mcc': MCC_coeff
	}
	
	if not binary_class:
		metrics.update(
			{
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
	)
	
	# - For binary class compute best TSS vs threshold
	if binary_class:
		# - probs from logits already computed above
		p_pos = probs[:, 1].numpy()          # assumes index 1 is positive (C+ or M+)
		y_true_np = y_true

		# - log and add to the returned metrics dict
		best = best_tss_from_probs(p_pos, y_true_np)
		metrics.update({
			"tss_best": best["tss"],
			"tss_best_thr": best["thr"],
			"tss_best_tpr": best["tpr"],
			"tss_best_tnr": best["tnr"],
		})
	
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
			"accuracy": float(metrics["accuracy"]),
			"precision": float(metrics["precision"]),
			"recall": float(metrics["recall"]),
			"f1score_weighted": float(metrics["f1score_weighted"]),
			"f1score_micro": float(metrics["f1score_micro"]),
			"f1score_macro": float(metrics["f1score_macro"]),
			"mcc": float(metrics["mcc"]),
		}
		
		# - Check if TP, FN etc are scalars
		if np.isscalar(metrics["fp"]):
			metrics_scalar.update(
				{
					"fp": int(metrics["fp"]),
					"fn": int(metrics["fn"]),
					"tp": int(metrics["tp"]),
					"tn": int(metrics["tn"]),
					"tpr": float(metrics["tpr"]),
					"tnr": float(metrics["tnr"]),
					"ppv": float(metrics["ppv"]),
					"npv": float(metrics["npv"]),
					"fpr": float(metrics["fpr"]),
					"fnr": float(metrics["fnr"]),
					"fdr": float(metrics["fdr"]),
					"tss": float(metrics["tss"]),
					"hss": float(metrics["hss"]),
					"gss": float(metrics["gss"]),
				}
			)
		
		# - Check if summary metrics are present (multiclass)
		if "tpr_summary" in metrics:
			metrics_scalar.update(
				{
					"tpr_macro": float(metrics["tpr_summary"]["macro"]),
					"tpr_weighted": float(metrics["tpr_summary"]["weighted"]),
					"tnr_macro": float(metrics["tnr_summary"]["macro"]),
					"tnr_weighted": float(metrics["tnr_summary"]["weighted"]),
					"hss_macro": float(metrics["hss_summary"]["macro"]),
					"tss_macro": float(metrics["tss_summary"]["macro"]),
					"gss_macro": float(metrics["gss_summary"]["macro"]),
					"hss_weighted": float(metrics["hss_summary"]["weighted"]),
					"tss_weighted": float(metrics["tss_summary"]["weighted"]),
					"gss_weighted": float(metrics["gss_summary"]["weighted"]),
					"tpr_micro": float(metrics["tpr_micro"]),
					"tnr_micro": float(metrics["tnr_micro"]),
					"hss_micro": float(metrics["hss_micro"]),
					"tss_micro": float(metrics["tss_micro"]),
					"gss_micro": float(metrics["gss_micro"]),
				}
			)
			
		return metrics_scalar
		
	return compute_single_label_metrics


##########################################
##   ORDINAL METRICS
##########################################
def ordinal_metrics_from_logits(predictions, labels, target_names=None, thresholds=None):
	"""
		Evaluate an ordinal (CORAL/cumulative) model.
			- predictions: np.ndarray or torch.Tensor, shape (B, K-1) logits for [>=class1 .. >=class_{K-1}]
			- labels: (B,) ints in {0..K-1} OR (B, K-1) cumulative 0/1 labels
			- returns: dict with your usual single-label metrics + ordinal extras
	"""
	# -- to torch
	logits_t = torch.as_tensor(predictions)
	B, Km1 = logits_t.shape
	K = Km1 + 1

	# -- normalize labels to class indices
	if isinstance(labels, torch.Tensor):
		lab = labels.detach().cpu().numpy()
	else:
		lab = np.asarray(labels)

	if lab.ndim == 2 and lab.shape[1] == Km1:
		# cumulative labels -> class index by counting ones
		y_true = lab.sum(axis=1).astype(int)
	else:
		y_true = lab.astype(int)  # already indices

	# -- build 4-class probabilities from CORAL (so we can reuse your single_label_metrics)
	probs4 = coral_probs_from_logits(logits_t)         # (B, K)
    
	# Pass "log-probs" so your single_label_metrics softmax -> original probs
	log_probs4 = torch.log(probs4)

	# -- get your standard dict (accuracy, F1s, cm, TSS/HSS/GSS, etc.)
	base = single_label_metrics(
		predictions=log_probs4.detach().cpu().numpy(),  # shape (B, K)
		labels=y_true,
		target_names=target_names
	)

	# -- ordinal-specific extras
	# decode classes directly (should match argmax of probs4)
	y_pred = coral_decode_classes_from_logits(logits_t, thresholds=thresholds)

	# Quadratic Weighted Kappa (ordinal agreement)
	try:
		qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
	except Exception:
		qwk = np.nan

	# per-threshold ROC-AUC / AP (treat each boundary as binary task)
	p_ge = torch.sigmoid(logits_t).detach().cpu().numpy()  # (B, K-1)
	aucs, aps = [], []
	for k in range(1, K):  # thresholds 1..K-1
		y_bin = (y_true >= k).astype(int)
		try:
			aucs.append(roc_auc_score(y_bin, p_ge[:, k-1]))
		except Exception:
			aucs.append(np.nan)
        
		try:
			aps.append(average_precision_score(y_bin, p_ge[:, k-1]))
		except Exception:
			aps.append(np.nan)

	mono = monotonicity_violations(logits_t)
	base.update({
		"qwk_quadratic": qwk,
		"auc_thresholds": aucs,           # list length K-1
		"ap_thresholds": aps,             # list length K-1
		"monotonicity_violation_rate": mono["mvr"],
		"monotonicity_violation_magnitude": mono["mvm"],
	})

	return base


def build_ordinal_metrics(target_names=None, thresholds=None):
	""" Returns a HF Trainer-compatible compute_metrics for ordinal models. """
    
	def compute_ordinal_metrics(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions  # (B, K-1)
		nonlocal target_names, thresholds

		full = ordinal_metrics_from_logits(
			predictions=preds,
			labels=p.label_ids,
			target_names=target_names,
			thresholds=thresholds
		)

		# Scalars that Trainer logs
		metrics_scalar = {
			"accuracy": full["accuracy"],
			"precision": full["precision"],
			"recall": full["recall"],
			"f1score_weighted": full.get("f1score_weighted", full.get("f1score", np.nan)),
			"f1score_micro": full.get("f1score_micro", np.nan),
			"f1score_macro": full.get("f1score_macro", np.nan),
			"mcc": full.get("mcc", np.nan),
			"tss_micro": full.get("tss_micro", np.nan),
			"hss_micro": full.get("hss_micro", np.nan),
			"gss_micro": full.get("gss_micro", np.nan),
			"qwk_quadratic": full["qwk_quadratic"],
			"mvr": full["monotonicity_violation_rate"],
			"mvm": full["monotonicity_violation_magnitude"],
		}

		# Add per-threshold AUC/AP with readable keys
		aucs = full["auc_thresholds"]
		aps  = full["ap_thresholds"]
		for i, (auc, ap) in enumerate(zip(aucs, aps), start=1):
			metrics_scalar[f"auc_ge_{i}"] = auc
			metrics_scalar[f"ap_ge_{i}"]  = ap

		return metrics_scalar

	return compute_ordinal_metrics


