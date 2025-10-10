import os
import sys
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

def draw_hmi(filename, save=False, cmap="gray"):
	""" Read & draw hmi """
	
	# - Read image
	image= Image.open(filename)

	# - Draw image
	#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

	plt.imshow(image, cmap="gray")
	plt.axis('off')    

	if save:
		plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
	else:
		plt.show()
		

def draw_hmi_video(filenames, save=False, cmap="gray"):
	""" Read & draw hmi """
	
	# - Draw image
	nframes= len(filenames)
	fig, ax = plt.subplots(nrows=1, ncols=nframes, figsize=(20, 8))

	# - Read images
	images= []
	for i in range(nframes):
		print(f"--> Drawing frame {filenames[i]} ...")
		image= Image.open(filenames[i])
		images.append(image)
		ax[i].imshow(image, cmap="gray")
		ax[i].axis('off')    

	# - Save/draw
	if save:
		plt.savefig('video.png', bbox_inches='tight', pad_inches=0)
	else:
		plt.show()


# - Read args
p = argparse.ArgumentParser(description="Parser for reading data")
p.add_argument("--metadata", required=True, type=str)
p.add_argument("--ar_sel", required=False, type=int, default=-1)
p.add_argument("--label_sel", required=False, type=str, default="")
p.add_argument("--index_sel", required=False, type=int, default=-1)
p.add_argument("--select_one_per_ar", action="store_true")
p.add_argument("--save", action="store_true")
args= p.parse_args()	

inputfile= args.metadata
label_sel= args.label_sel
index_sel= args.index_sel
save= args.save

# - Read metadata
print(f"Reading metadata {inputfile} ...")
f= open(inputfile, "r")
d= json.load(f)["data"]

if index_sel!=-1:
	if index_sel>=len(d):
		print(f"ERROR: Sel index exceeding data size ({index_sel})!")
		sys.exit(1)

# - Detect if image/video
is_video= False
if 'filepaths' in d[0]:
	is_video= True 

# - Extract selected data by label
if label_sel=="":
	#d_sel= d
	indices= [idx for idx, item in enumerate(d) if item["label"]==label_sel]
else:
	print(f"Extracting data for label {label_sel} ...")
	indices= [idx for idx, item in enumerate(d) if item["label"]==label_sel]
	#d_sel= [d[index] for index in indices]

# - Extract selected data by AR
if args.ar_sel!=-1:
	print(f"Extracting data for AR {args.ar_sel} ...")
	indices= [idx for idx in indices if d[idx]["ar"]==args.ar_sel]
	
# - Get list of ARs
unique_ars= list(set([d[index]["ar"] for index in indices]))
print("unique_ars")
print(unique_ars)
index_ar_dict= {}
for index in indices:
	ar= d[index]["ar"]
	index_ar_dict[ar]= index
	
if args.select_one_per_ar:
	indices= list(index_ar_dict.values())

# - Select index of data to be read
if index_sel!=-1:
	if is_video:
		print(f"--> Drawing item no. {index_sel} ...")
		filenames= d[index_sel]["filepaths"]
		draw_hmi_video(filenames, args.save, cmap="gray")
	else:
		filename= d[index_sel]["filepath"]
		print(f"--> Drawing item no. {index_sel}: {filename} ...")
		draw_hmi(filename, args.save, cmap="gray")

else:
	# - Reading data in loop
	if is_video:
		for index in indices:
			filenames= d[index]["filepaths"]
			label= d[index]["label"]
			ar= d[index]["ar"]
			print(f"--> Drawing item no. {index} (AR={ar}, label={label}): {str(filenames)} ...")
			draw_hmi_video(filenames, args.save, cmap="gray")
	else:
		for index in indices:
			filename= d[index]["filepath"]
			label= d[index]["label"]
			ar= d[index]["ar"]
			print(f"--> Drawing item no. {index} (AR={ar}, label={label}): {filename} ...")
			draw_hmi(filename, args.save, cmap="gray")
	
