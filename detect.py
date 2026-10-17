# detect.py - find objects in an image using a text prompt

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 1. Choose the model and the device (GPU if available)
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the model (first time: downloads ~700 MB)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

# 3. Load the image and write what we are looking for
image = Image.open("images/cats.jpg").convert("RGB")
text = "a cat. a remote control."   # lowercase, each object ends with a dot

# 4. Prepare the inputs and run the model
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():               # no training, only prediction
    outputs = model(**inputs)

# 5. Turn the raw output into boxes, scores and labels
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,         # how sure the model must be about the box
    text_threshold=0.3,    # how sure it must be about the label
    target_sizes=[image.size[::-1]],  # (height, width) of the original image
)[0]

# 6. Print what was found
labels = results["text_labels"]
for box, score, label in zip(results["boxes"], results["scores"], labels):
    x1, y1, x2, y2 = [round(v) for v in box.tolist()]
    print(f"{label:15s} score={score:.2f}  box=({x1}, {y1}, {x2}, {y2})")