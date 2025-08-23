#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
# - STANDARD MODULES
import os
import sys
import json
import numpy as np
import argparse

# - SKLEARN
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.metrics import matthews_corrcoef

###########################
##     ARGS
###########################
def get_args():
	"""This function parses and return arguments passed in"""
	parser = argparse.ArgumentParser(description="Parse args.")

	# - Input options
	parser.add_argument('-inputfile','--inputfile', dest='inputfile', required=True, type=str, help='Input prediction json file') 
	
	parser.add_argument('--binary', dest='binary', action='store_true',help='Choose binary classification label scheme (default=false)')	
	parser.set_defaults(binary=False)
	parser.add_argument('-flare_thr', '--flare_thr', dest='flare_thr', required=False, type=str, default='C', action='store',help='Choose flare class label name: {C-->label=C+,M-->label=M+}.')
	
	args = parser.parse_args()	

	return args		


#################
##  FUNCTIONS
#################
LABEL_REMAP= {
	"NONE": "NONE",
	"C": "C",
	"M": "M",
	"X": "X"
}

LABEL2ID= {
	"NONE": 0,
	"C": 1,
	"M": 2,
	"X": 3
}

LABEL2ID_BINARY_CTHR= {
	"NONE": 0,
	"C": 1,
	"M": 1,
	"X": 1
}

LABEL2ID_BINARY_MTHR= {
	"NONE": 0,
	"C": 0,
	"M": 1,
	"X": 1
}

ID_REMAP= {
	0: 0,
	1: 1,
	2: 2,
	3: 3
}

LABEL_REMAP_BINARY_CTHR= {
	"NONE": "NONE",
	"C": "C+",
	"M": "C+",
	"X": "C+"
}

ID_REMAP_BINARY_CTHR= {
	0: 0,
	1: 1,
	2: 1,
	3: 1
}

LABEL_REMAP_BINARY_MTHR= {
	"NONE": "NONE",
	"C": "NONE",
	"M": "M+",
	"X": "M+"
}

ID_REMAP_BINARY_MTHR= {
	0: 0,
	1: 0,
	2: 1,
	3: 1
}


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)       
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return super().default(obj)
		
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

def compute_single_label_metrics(y_true, y_pred, target_names=None):
	""" Compute single-label metrics """
	
	# If target names are provided, infer number of classes from them
	if target_names is not None:
		num_classes = len(target_names)
		label_indices = list(range(num_classes))
	else:
		# Fallback: infer from y_true and y_pred
		label_indices = sorted(set(y_true).union(set(y_pred)))
        
	# - Compute classification report
	class_report= classification_report(
		y_true, 
		y_pred, 
		labels=label_indices,
		target_names=target_names if target_names else None,
		output_dict=True
	)
	
	# - Compute acc/prec/recall/F1 metrics
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
	
	
	return metrics


##############
##   MAIN   ##
##############
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	print("Get script args ...")
	try:
		args= get_args()
	except Exception as ex:
		print(f"Failed to get and parse options (err={str(ex)})")
		return 1
		
	label2id= LABEL2ID
	if args.binary:
		if args.flare_thr=="C":
			label2id= LABEL2ID_BINARY_CTHR
		elif args.flare_thr=="M":
			label2id= LABEL2ID_BINARY_MTHR
		else:
			print(f"ERROR: Invalid/unsupported flare_thr {args.flare_thr}!")
			return 1
		
	#===========================
	#==   READ INPUTFILE
	#===========================
	print(f"INFO: Reading file {args.inputfile} ...")
	f= open(args.inputfile, "r")
	d= json.load(f)["data"]
	
	print(f"INFO: #{len(d)} data read ...")
	
	# - Compute y_true & y_pred
	print("INFO: Computing y_true & y_pred from file ...")
	ids= []
	ids_pred= []
	
	for item in d:
		label= item["label"]
		label_pred= item["label_pred"]
		id= label2id[label]
		id_pred= label2id[label_pred]
		ids.append(id)
		ids_pred.append(id_pred)
		
	y_true= np.array(ids)
	y_pred= np.array(ids_pred)
	
	#===========================
	#==   COMPUTE METRICS
	#===========================
	print("INFO: Computing metrics ...")
	metrics= compute_single_label_metrics(y_true, y_pred)
		
	print("--> metrics")
	print(metrics)
	
	metrics_pretty= json.dumps(metrics, indent=2, cls=NumpyEncoder)
	print(metrics_pretty)
		
	return 0

###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
