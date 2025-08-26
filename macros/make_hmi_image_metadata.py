import os
import json
import argparse
import pandas as pd
import re
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
##  PARSE LABELS
##############################
def parse_label(label):
	""" Parse label """
	if str(label).strip() in {"0", "NONE", "", "nan"}:
		return "NONE"
	if str(label)[0] in {"C", "M", "X"}:
		return str(label)[0]
	return "NONE"

##############################
##  LOAD IMAGE LABEL FILE
##############################
def load_image_labels(label_file):
	""" Load image label file """
    
	df = pd.read_csv(label_file, comment="#", names=["ImgName", "Label"], dtype={'Label': str})

	label_map = {}
	label_map_orig= {}
	for _, row in df.iterrows():
		fname= row["ImgName"]
		label= row["Label"]
		label_map[fname] = parse_label(label)
		label_map_orig[fname] = label
	
	return label_map, label_map_orig

def get_sname(filepath):
	return Path(filepath).stem

##############################
##  PARSE/CHECK TIMESTAMPS
##############################
def parse_timestamp_from_filename(filename):
	""" Parse timestamp """
	
	match = re.search(r"\.(\d{8})_(\d{6})_", filename)
	if not match:
		raise ValueError(f"Could not extract timestamp from: {filename}")
	date_str, time_str = match.groups()
	return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

def validate_timing(ar_groups, cadence_minutes=12):
	""" Validate timing among images """
	
	for ar, images in ar_groups.items():
		for i in range(1, len(images)):
			t1 = parse_timestamp_from_filename(images[i - 1])
			t2 = parse_timestamp_from_filename(images[i])
			delta = (t2 - t1).total_seconds() / 60
			if delta != cadence_minutes:
				print(f"WARN: AR {ar}, images {images[i - 1]} and {images[i]} are {delta} min apart")

def compute_img_timediff(imgname1, imgname2):
	t1 = parse_timestamp_from_filename(imgname1)
	t2 = parse_timestamp_from_filename(imgname2)
	delta_t = (t2 - t1).total_seconds() / 60
	return delta_t

##############################
##  GENERATE METADATA
##############################
def generate_metadata_single_horizon(args):
	""" Generate metadata """

	# - Set variables
	root_dir= args.root_dir
	inputfile_label= args.inputfile_label 
	inputfile_ar= args.inputfile_ar
	outfile_metadata= args.outfile_metadata
	outfile_label= args.outfile_label
	forecast_offset = args.forecast_offset
	cadence_minutes= args.cadence_minutes 

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

	# - Read image label file
	print("INFO: Read image label file ...")
	labels, labels_orig = load_image_labels(inputfile_label)
	ar_groups = defaultdict(list)

	# - Read AR list file
	print("INFO: Read AR list file ...")
	with open(inputfile_ar, "r") as f:
		train_ars = set(line.strip() for line in f if not line.startswith("#"))

	for imgname in labels:
		parts = imgname.split("_")
		ar = parts[0].replace("AR", "")
		if ar in train_ars:
			ar_groups[ar].append(imgname)  # preserve original CSV order 

	# - Sort image filenames chronologically within each AR
	print("INFO: Sort image filenames chronologically within each AR ...")
	for ar in ar_groups:
		ar_groups[ar].sort(key=parse_timestamp_from_filename)

	# - Validate timing in file
	print("INFO: Validate timing in file ...")
	validate_timing(ar_groups, cadence_minutes)

	# - Compute metadata    
	metadata = {"data": []}

	
	if forecast_offset == 24:
		#====================================
		#==     PREDEFINED HORIZON
		#====================================
		for ar, images in ar_groups.items():
			for imgname in images:
				print(f"Processing file {imgname} ...")
				imgname_fullpath= os.path.join(args.root_dir, ar, imgname)
				timestamp= parse_timestamp_from_filename(imgname)
				label = labels[imgname]
				if label not in {"NONE", "C", "M", "X"}:
					print(f"WARN: Label {label} not recognized, skip image {imgname} ...")
					continue
		
				entry = {
					"filepath": imgname_fullpath,
					"sname": get_sname(imgname),
					"label": label_remap[label],
					"id": label2id[label],
					"ar": int(ar),
					"timestamp": str(timestamp),
					"flare_type": label,
					"flare_id": LABEL2ID[label]
				}
				metadata["data"].append(entry)

	else:
		#==================================================
		#==     CUSTOM HORIZON (slow & not fully working)
		#==================================================
		# NB: This is slow and not fully working. It would be better to produce again the label file
		#     from flare event file, but the provided event file seems not complete, as I cannot reproduce the original label file.
		effective_offset = forecast_offset - 24
		if effective_offset < 0:
			raise ValueError("Forecast offset must be >= 24h due to label structure.")
		steps_forward = effective_offset * 60 // cadence_minutes

		for ar, images in ar_groups.items():
			print(f"--> {ar}: no {len(images)}, step_forward={steps_forward}")

			# - Loop over images in this AR
			for i in range(len(images)):
				#future_index= i + steps_forward
				#if future_index>=len(images):
				#	print(f"Skipping {images[i]}: insufficient frames for {forecast_offset}h forecast")
				#	continue

				# - Search future images at desired timestamp diff
				current_img = images[i]
				future_img= None
				flare_labels_until_forecast_point= []
				for j in range(i+1, len(images)):
					label_curr = parse_label(labels[images[j]])
					flare_labels_until_forecast_point.append(label_curr)

					# - Check time diff
					tdiff= compute_img_timediff(current_img, images[j]) # in minutes
					if tdiff==effective_offset*60:
						future_img= images[j]
						break

				# - Check if future image was found
				if future_img is None:
					print(f"Skipping {current_img} as did not find any future image at the desired time offset {effective_offset} ...")
					continue

				#future_img = images[future_index]
				if future_img not in labels:
					print(f"Skipping {current_img}: no label for future image {future_img}")
					continue

				# - Check img & future_img time diff
				tdiff= compute_img_timediff(current_img, future_img) # in minutes
				if tdiff!=effective_offset*60:
					print(f"Skipping {current_img} as {future_img} timediff is {tdiff} min (expected={effective_offset*60}) ...")
					continue

				# - Check how many flare labels are found until forecasting future point (it should be only one class?)
				flare_label_set= set(flare_labels_until_forecast_point)
				n_labels= len(flare_label_set)
				if n_labels>1: 
					print(f"WARN: For {current_img} found {n_labels} labels until forecasting point. Labels: {flare_label_set}")

				# - Save label data to dict
				#label = parse_label(labels.get(future_img, "0"))
				label = parse_label(labels[future_img])
				if label not in {"NONE", "C", "M", "X"}:
					print(f"WARN: Label {label} not recognized, skip image {imgname} ...")
					continue

				imgname_fullpath= os.path.join(args.root_dir, ar, images[i])

				entry = {
					"filepath": imgname_fullpath,
					"filepath_future": future_img,
					"sname": get_sname(images[i]),
					"label": label_remap[label],
                                        "id": label2id[label],
					"ar": int(ar),
					"flare_type": label,
                                        "flare_id": LABEL2ID[label]
				}
				metadata["data"].append(entry)

	# - Write metadata
	with open(outfile_metadata, "w") as f:
		json.dump(metadata, f, indent=2)

	# - Write sorted CSV if requested
	with open(outfile_label, "w") as f:
		f.write("# ImgName,Label\n")
		for ar, images in ar_groups.items():
			for img in images:
				f.write(f"{img},{labels_orig[img]}\n")

	
###############################
###        MAIN
###############################
if __name__ == "__main__":

	# - Define & parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--inputfile_label", required=True, help="Path to label file (e.g. C1.0_24hr_224_png_Labels.txt)")
	parser.add_argument("--inputfile_ar", required=True, help="Path to file listing ARs to include")
	parser.add_argument("--outfile_metadata", required=False, default="metadata.json", help="Path to output metadata JSON")
	parser.add_argument("--cadence_minutes", type=int, default=12, help="Time between images in minutes")
	parser.add_argument("--forecast_offset", type=int, default=24, help="Forecast horizon in hours")
	parser.add_argument("--root_dir", type=str, required=True, help="Root directory where AR image folders are stored")
	parser.add_argument("--outfile_label", type=str, help="Path to write sorted CSV with labels")

	parser.add_argument('--binary', dest='binary', action='store_true',help='Choose binary classification label scheme (default=false)')	
	parser.set_defaults(binary=False)
	parser.add_argument('-flare_thr', '--flare_thr', dest='flare_thr', required=False, type=str, default='C', action='store',help='Choose flare class label name: {C-->label=C+,M-->label=M+}.')

	args = parser.parse_args()

	# - Run 	
	generate_metadata_single_horizon(args)
	
