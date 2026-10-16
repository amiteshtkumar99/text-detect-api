# run_detect.py - run the detector on one image from the terminal

import argparse
import os
import time

import cv2
from PIL import Image

from detector import TextDetector, draw_detections

# 1. Read the inputs from the command line
parser = argparse.ArgumentParser(description="Text-prompted object detection")
parser.add_argument("image", help="path to the input image")
parser.add_argument("prompt", help='objects to find, e.g. "a cat. a remote control."')
parser.add_argument("--threshold", type=float, default=0.4, help="minimum box score")
parser.add_argument("--out", default="outputs", help="folder for the result image")
args = parser.parse_args()

# 2. Build the machine (loads the model once)
detector = TextDetector()

# 3. Detect and measure the time
image = Image.open(args.image).convert("RGB")
start = time.time()
detections = detector.detect(image, args.prompt, box_threshold=args.threshold)
ms = (time.time() - start) * 1000

for d in detections:
    print(d)
print(f"Found {len(detections)} objects in {ms:.0f} ms")

# 4. Draw and save the result
os.makedirs(args.out, exist_ok=True)
name = os.path.splitext(os.path.basename(args.image))[0]
out_path = os.path.join(args.out, f"{name}_result.jpg")
cv2.imwrite(out_path, draw_detections(image, detections))
print(f"Saved: {out_path}")