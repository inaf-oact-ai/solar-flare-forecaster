import os
import json
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta

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

LABEL_REMAP= {
	"NONE": "NONE",
	"C": "C",
	"M": "M",
	"X": "X"
}

LABEL_REMAP_BINARY_CTHR= {
	"NONE": "NONE",
	"C": "C+",
	"M": "C+",
	"X": "C+"
}

LABEL_REMAP_BINARY_MTHR= {
	"NONE": "NONE",
	"C": "NONE",
	"M": "M+",
	"X": "M+"
}


##############################
##  GENERATE METADATA
##############################
def generate_metadata(args):
	""" Generate metadata """

	# - Set variables
	label2id= LABEL2ID
	label_remap= LABEL_REMAP
	if args.binary:
		if args.flare_thr=="C":
			label2id= LABEL2ID_BINARY_CTHR
			label_remap= LABEL_REMAP_BINARY_CTHR
		elif args.flare_thr=="M":
			label2id= LABEL2ID_BINARY_MTHR
			label_remap= LABEL_REMAP_BINARY_MTHR

	print("label2id")
	print(label2id)
	print("label_remap")
	print(label_remap)


	# - Read metadata
	f= open(args.inputfile, "r")
	d= json.load(f)["data"]
	
	# - Modify metadata labels
	metadata = {"data": []}

	for item in d:
		label= item["label"]

		entry = {
			"filepath": item["filepath"],
			"sname": item["sname"],
			"label": label_remap[label],
			"id": label2id[label],
			"ar": item["ar"],
			"timestamp": item["timestamp"],
			"flare_type": item["flare_type"],
			"flare_id": item["flare_id"]
		}
		metadata["data"].append(entry)

	# - Write metadata
	with open(args.outfile, "w") as f:
		json.dump(metadata, f, indent=2)


###############################
###        MAIN
###############################
if __name__ == "__main__":

	# - Define & parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--inputfile", required=True, help="Path to metadata file")
	parser.add_argument("--outfile", type=str, help="Path to output metadata")
	
	parser.add_argument('--binary', dest='binary', action='store_true',help='Choose binary classification label scheme (default=false)')	
	parser.set_defaults(binary=False)
	parser.add_argument('-flare_thr', '--flare_thr', dest='flare_thr', required=False, type=str, default='C', action='store',help='Choose flare class label name: {C-->label=C+,M-->label=M+}.')

	args = parser.parse_args()
	
	# - Run 	
	generate_metadata(args)
