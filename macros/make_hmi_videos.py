import os
import cv2
import json
import argparse
import pandas as pd
from pathlib import Path

# Define command-line arguments
parser = argparse.ArgumentParser(description="Create AR flare forecast videos and metadata.")
parser.add_argument("--input_labels", type=str, required=True, help="Path to input labels file")
parser.add_argument("--train_ars", type=str, required=True, help="Path to CSV file listing ARs to process")
parser.add_argument("--output_dir", type=str, default="output_videos", help="Directory to save videos and metadata")
parser.add_argument("--video_length", type=int, default=16, help="Number of frames per video")
parser.add_argument("--frame_step_minutes", type=int, default=72, help="Step between frames in minutes")
parser.add_argument("--cadence", type=int, default=12, help="Cadence between images in minutes")
parser.add_argument("--skip_incomplete", action="store_true", help="Skip videos with fewer than video_length frames")
parser.add_argument("--create_videos", action="store_true", help="Actually create video files")
parser.add_argument("--root_dir", type=str, required=True, help="Root directory where AR image folders are stored")
parser.add_argument("--video_format", type=str, choices=["mp4", "avi"], default="mp4", help="Output video format")
parser.add_argument("--lossless", action="store_true", help="Use lossless video codec if available")
args = parser.parse_args()

# Derived values
frame_step = args.frame_step_minutes // args.cadence
horizons_minutes = {"label_24": 24 * 60, "label_36": 36 * 60, "label_48": 48 * 60}
horizon_frames = {key: val // args.cadence for key, val in horizons_minutes.items()}

# Flare label priority
flare_priority = {'NONE': 0, 'C': 1, 'M': 2, 'X': 3}
label2id = {'NONE': 0, 'C': 1, 'M': 2, 'X': 3}

# Ensure output directory exists
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# Load labels
labels_df = pd.read_csv(args.input_labels, comment='#', header=None, names=["ImgName", "Label"], dtype={'Label': str})
labels_df['AR'] = labels_df['ImgName'].apply(lambda x: x.split('_')[0])

# Normalize labels
def normalize_label(label):
    if label == '0':
        return 'NONE'
    return label[0]

labels_df['NormLabel'] = labels_df['Label'].apply(normalize_label)
grouped_by_ar = labels_df.groupby('AR')

# Load train AR list
train_ars = pd.read_csv(args.train_ars, comment='#', header=None, names=['AR'])['AR'].astype(str).tolist()

# Codec and file extension
if args.video_format == "avi":
    fourcc = cv2.VideoWriter_fourcc(*('FFV1' if args.lossless else 'MJPG'))
    ext = "avi"
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ext = "mp4"

print(f"Video output format: {ext}")

# Initialize metadata
metadata = {"data": []}

# Process each AR
for ar in train_ars:
    print(f"--> Processing AR {ar} ...")
    if ar not in grouped_by_ar.groups:
        print(f"WARN: AR {ar} not in group, skipping ...")
        continue

    ar_group = grouped_by_ar.get_group(ar).reset_index(drop=True)
    images = ar_group['ImgName'].tolist()
    norm_labels = ar_group['NormLabel'].tolist()

    for start_idx in range(len(images)):
        frame_indices = [start_idx + i * frame_step for i in range(args.video_length)]
        if frame_indices[-1] >= len(images):
            if args.skip_incomplete:
                continue
            frame_indices = [idx for idx in frame_indices if idx < len(images)]

        frame_names = [images[i] for i in frame_indices]
        frame_paths = [os.path.join(args.root_dir, ar, name) for name in frame_names]
        last_frame_idx = frame_indices[-1]

        # Compute flare labels for each horizon
        label_dict = {}
        for label_key, horizon in horizon_frames.items():
            search_end = last_frame_idx + horizon
            search_range = norm_labels[last_frame_idx + 1: search_end + 1]
            max_label = 'NONE'
            for l in search_range:
                if flare_priority[l] > flare_priority[max_label]:
                    max_label = l
            label_dict[label_key] = max_label

        frame_files = []
        video_path = os.path.join(args.output_dir, f"{ar}_{start_idx}.{ext}")
        out = None

        if args.create_videos:
            print(f"--> Creating video file {video_path} ...")
            for img_path in frame_paths:
                if not os.path.isfile(img_path):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                height, width, _ = img.shape
                if out is None:
                    out = cv2.VideoWriter(video_path, fourcc, 1, (width, height))
                out.write(img)
                frame_files.append(img_path)
            if out:
                out.release()
        else:
            frame_files = frame_paths

        print(f"AR {ar}: Video files: %s" % (str(frame_files)))

        #labels= [label_dict["label_24"], label_dict["label_36"], label_dict["label_48"]]

        if args.create_videos:
            metadata['data'].append({
                "filepath": video_path,
                "frames": frame_files,
                "id": label2id[labels[0]],
                "label": labels[0],
                #"ids": [label2id[l] for l in labels],
                #"labels": labels,
                "ar": ar
            })
        else:
            metadata['data'].append({
                "filepaths": frame_files,
                "id": label2id[labels[0]],
                "label": labels[0],
                "ar": ar
            })

# Save metadata
metadata_path = os.path.join(args.output_dir, "metadata.json")
print(f"--> Saving metadata file {metadata_path} ...")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
