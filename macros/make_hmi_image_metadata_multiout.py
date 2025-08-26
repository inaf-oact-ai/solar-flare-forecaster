import os
import json
import argparse
import pandas as pd
from collections import defaultdict

def parse_label(label):
    if label == "0" or label == 0:
        return "NONE"
    return str(label)[0]

def load_image_labels(label_file):
    df = pd.read_csv(label_file, comment="#", names=["ImgName", "Label"])
    label_map = {}
    for _, row in df.iterrows():
        label_map[row["ImgName"]] = parse_label(row["Label"])
    return label_map

def generate_metadata(input_labels, train_ars_file, output_json, cadence_minutes=12, forecast_offsets=[12, 36, 48]):
    labels = load_image_labels(input_labels)
    ar_groups = defaultdict(list)

    with open(train_ars_file, "r") as f:
        train_ars = set(line.strip() for line in f if not line.startswith("#"))

    for imgname in labels:
        parts = imgname.split("_")
        ar = parts[0].replace("AR", "")
        if ar in train_ars:
            ar_groups[ar].append(imgname)

    for ar in ar_groups:
        ar_groups[ar] = sorted(ar_groups[ar], key=lambda name: name.split(".")[1])

    metadata = {"data": []}
    step_per_forecast = {f: int(f // cadence_minutes) for f in forecast_offsets}

    for ar, images in ar_groups.items():
        for i in range(len(images)):
            entry = {
                "filepath": images[i],
                "labels": [],
                "ids": [],
            }
            for f in forecast_offsets:
                idx = i + step_per_forecast[f]
                if idx < len(images):
                    future_img = images[idx]
                    label = parse_label(labels[future_img])
                else:
                    label = "NONE"
                entry["labels"].append(label)
                entry["ids"].append({"NONE": 0, "C": 1, "M": 2, "X": 3}[label])
            metadata["data"].append(entry)

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_labels", required=True, help="Path to C1.0_24hr_224_png_Labels.txt")
    parser.add_argument("--train_ars", required=True, help="Path to file listing ARs to include")
    parser.add_argument("--output_json", required=True, help="Path to output metadata JSON")
    parser.add_argument("--cadence_minutes", type=int, default=12, help="Time between images in minutes")
    parser.add_argument("--forecast_offsets", type=int, nargs="+", default=[12, 36, 48], help="Forecast horizons in hours")
    args = parser.parse_args()

    forecast_offsets = [f * 60 for f in args.forecast_offsets]
    generate_metadata(args.input_labels, args.train_ars, args.output_json, args.cadence_minutes, forecast_offsets)
