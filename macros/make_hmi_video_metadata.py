import os
import json
import argparse
import re
#import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# Flare label
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

def validate_timing_in_ar_group(ar_groups, cadence_minutes=12):
	""" Validate timing among images in AR groups """
	
	status= True
	for ar, images in ar_groups.items():
		for i in range(1, len(images)):
			try:
				t1 = parse_timestamp_from_filename(images[i - 1])
				t2 = parse_timestamp_from_filename(images[i])
			except Exception as e:
				print(f"WARN: Exception (err={str(e)}) occurred when computing timestamps of images: {images[i - 1]}, {images[i]}")
				continue
			
			delta = (t2 - t1).total_seconds() / 60
			if delta != cadence_minutes:
				print(f"WARN: AR {ar}, images {images[i - 1]} and {images[i]} are {delta} min apart")
				status= False
				
	return status
	
def validate_timing(images, cadence_minutes=12):
	""" Validate timing among images """
	
	# - Compute timestamps from images
	try:
		timestamps= [parse_timestamp_from_filename(item) for item in images]
	except Exception as e:
		print("WARN: Failed to compute timestamps of given image frames (err=%s), returning failed check!" % (str(e)))
		return False
	
	# - Validate timestamps
	status= True
	
	for i in range(len(timestamps)-1):
		t = timestamps[i]
		t_next = timestamps[i+1]
		
		delta = (t_next - t).total_seconds() / 60
		if delta != cadence_minutes:
			print(f"WARN: images {images[i]} and {images[i+1]} are {delta} min apart")
			status= False
			break
				
	return status

	
def check_strictly_sorted(images):
	""" Check that images within each AR group are sorted in increasing time. """
	
	# - Compute timestamps	
	timestamps = []
	try:
		timestamps = [parse_timestamp_from_filename(img) for img in images]
	except Exception as e:
		print(f"WARN: Failed to parse timestamps in AR {ar} (err={str(e)})")
		return False

	# - Check sorting
	all_ok = True
	for i in range(1, len(timestamps)):
		if timestamps[i] <= timestamps[i - 1]:
			print(f"ERROR: Timestamps not strictly increasing at index {i}:")
			print(f"  {images[i-1]} → {timestamps[i-1]}")
			print(f"  {images[i]} → {timestamps[i]}")
			all_ok = False
			break

	return all_ok	
	
##############################
##  GENERATE METADATA
##############################
def generate_metadata(args):
	""" Generate metadata """
	
	# - Set options
	frame_step = args.frame_step_minutes // args.cadence_minutes
	#horizons_minutes = {"label_24": 24 * 60, "label_36": 36 * 60, "label_48": 48 * 60}
	#horizon_frames = {key: val // args.cadence for key, val in horizons_minutes.items()}

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
	print(f"INFO: Read image label file {args.inputfile_label} ...")
	label_map, label_map_orig = load_image_labels(args.inputfile_label)
	ar_groups = defaultdict(list)

	# - Read AR list file
	print(f"INFO: Read AR list file {args.inputfile_ar} ...")
	with open(args.inputfile_ar, "r") as f:
		train_ars = set(line.strip() for line in f if not line.startswith("#"))

	for imgname in label_map:
		parts = imgname.split("_")
		ar = parts[0].replace("AR", "")
		if ar in train_ars:
			ar_groups[ar].append(imgname)  # preserve original CSV order

	# - Sort image filenames chronologically within each AR
	print("INFO: Sort image filenames chronologically within each AR ...")
	for ar in ar_groups:
		ar_groups[ar].sort(key=parse_timestamp_from_filename)

	# - Validate timing in file
	print("INFO: Validate timing in images inside each AR group ...")
	validate_timing_in_ar_group(ar_groups, args.cadence_minutes)
	
	# - Generate metadata
	print("INFO: Generating metadata ...")
	metadata = {"data": []}
	
	for ar, images in ar_groups.items():
		print(f"--> {ar}: no {len(images)}, frame_step={frame_step}")
		
		# - Check images are indeed sorted in time
		sorting_ok= check_strictly_sorted(images)
		if not sorting_ok:
			print(f"Images in AR {ar} are not sorted in time, this is not expected, exit!")
			sys.exit(1)

		for start_idx in range(len(images)):
			if start_idx % 100 == 0:
				print(f"--> Processing index {start_idx}/{len(images)} ...")
    
			# - Compute frame indices
			frame_indices = [start_idx + i * frame_step for i in range(args.video_length)]
			
			first_frame_idx= frame_indices[0]
			first_frame_image= ""
			if first_frame_idx<len(images):
				first_frame_image= images[first_frame_idx]
			
			# - Check if frame indices exceed number of images
			#is_incomplete_sample= any( np.array(frame_indices) >= len(images) )
			is_incomplete_sample = any(i >= len(images) for i in frame_indices)
			if is_incomplete_sample:
				print(f"WARN: Skipping incomplete sample starting from image {first_frame_image} ...")
				continue
			
			# - Check that all frames are separated in time by frame_step_minutes
			frame_names = [images[i] for i in frame_indices]
			time_sep_ok= validate_timing(frame_names, args.frame_step_minutes)
			if not time_sep_ok:
				print(f"WARN: One/more frames in video are not separated by {args.frame_step_minutes} minutes, skipping video ...")
				continue	
        
			# - Full path names for all frames
			frame_paths = [os.path.join(args.root_dir, ar, name) for name in frame_names]
			
			# - Set label to last frame label
			last_frame_idx = frame_indices[-1]
			last_frame_image= images[last_frame_idx]
			label= label_map[last_frame_image]
			
			# - Get start & end timestamps
			t_start= parse_timestamp_from_filename(first_frame_image)
			t_end= parse_timestamp_from_filename(last_frame_image)
			
			# - Fill metadata
			metadata['data'].append({
				"filepaths": frame_paths,
				#"id": LABEL2ID[label],
				#"label": label,
				"label": label_remap[label],
				"id": label2id[label],
				"ar": int(ar),
				"t_start": str(t_start),
				"t_end": str(t_end),
				"flare_type": label,
				"flare_id": LABEL2ID[label]
			})

	# - Print final data sample size
	nsamples= len(metadata['data'])
	print(f"INFO: {nsamples} data generated ...") 

	# - Save metadata
	print(f"--> Saving metadata file {args.outfile_metadata} ...")
	with open(args.outfile_metadata, "w") as f:
		json.dump(metadata, f, indent=2)

###############################
###        MAIN
###############################
if __name__ == "__main__":

	# - Define & parse arguments
	parser = argparse.ArgumentParser(description="Create AR flare forecast videos and metadata.")
	parser.add_argument("--inputfile_label", type=str, required=True, help="Path to input labels file")
	parser.add_argument("--inputfile_ar", type=str, required=True, help="Path to CSV file listing ARs to process")
	parser.add_argument("--outfile_metadata", required=False, default="metadata.json", help="Path to output metadata JSON")
	parser.add_argument("--video_length", type=int, default=16, help="Number of frames per video")
	parser.add_argument("--frame_step_minutes", type=int, default=72, help="Step between frames in minutes")
	parser.add_argument("--cadence_minutes", type=int, default=12, help="Cadence between images in minutes")
	parser.add_argument("--root_dir", type=str, required=True, help="Root directory where AR image folders are stored")
	parser.add_argument('--binary', dest='binary', action='store_true',help='Choose binary classification label scheme (default=false)')    
	parser.set_defaults(binary=False)
	parser.add_argument('-flare_thr', '--flare_thr', dest='flare_thr', required=False, type=str, default='C', action='store',help='Choose flare class label name: {C-->label=C+,M-->label=M+}.')

	args = parser.parse_args()

	# - Generate metadata
	generate_metadata(args)
