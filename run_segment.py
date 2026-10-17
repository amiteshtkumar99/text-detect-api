# run_segment.py - detect with Grounding DINO, then segment with SAM

import argparse
import os
import time

import cv2
from PIL import Image

from detector import TextDetector
from segmenter import BoxSegmenter, draw_masks, mask_geometry

parser = argparse.ArgumentParser(description="Grounded-SAM: text -> boxes -> masks")
parser.add_argument("image")
parser.add_argument("prompt")
parser.add_argument("--threshold", type=float, default=0.3)
args = parser.parse_args()

detector = TextDetector()
segmenter = BoxSegmenter()
image = Image.open(args.image).convert("RGB")

# Run twice: the first run includes GPU warm-up, the second shows the real speed
for run in range(2):
    t0 = time.perf_counter()
    detections = detector.detect(image, args.prompt, box_threshold=args.threshold)
    t1 = time.perf_counter()
    masks = segmenter.segment(image, [d["box"] for d in detections])
    t2 = time.perf_counter()

print(f"Detection: {(t1 - t0) * 1000:.0f} ms | Segmentation: {(t2 - t1) * 1000:.0f} ms\n")

for d, mask in zip(detections, masks):
    g = mask_geometry(mask)
    x1, y1, x2, y2 = d["box"]
    box_centre = ((x1 + x2) // 2, (y1 + y2) // 2)
    print(f'{d["label"]:18s} score={d["score"]:.2f}  box_centre={box_centre}  '
          f'mask_centre={tuple(g["centre"])}  angle={g["angle_deg"]}°  area={g["area_px"]} px')

os.makedirs("outputs", exist_ok=True)
name = os.path.splitext(os.path.basename(args.image))[0]
out_path = f"outputs/{name}_masks.jpg"
cv2.imwrite(out_path, draw_masks(image, detections, masks))
print(f"\nSaved: {out_path}")