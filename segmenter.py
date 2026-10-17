# segmenter.py - turn boxes into exact object masks with SAM

import cv2
import numpy as np
import torch
from transformers import SamModel, SamProcessor

SAM_ID = "facebook/sam-vit-base"   # ~375 MB, fits easily next to Grounding DINO on 6 GB

COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0)]


class BoxSegmenter:
    """Loads SAM once. Given boxes, returns one mask per box."""

    def __init__(self, model_id=SAM_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM on {self.device} ...")
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def segment(self, image, boxes):
        """image: PIL RGB image. boxes: list of [x1, y1, x2, y2].
        Returns a list of masks (True/False arrays, same size as the image)."""
        if not boxes:
            return []

        inputs = self.processor(image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)   # one mask per box

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]                                    # shape: (num_boxes, 1, height, width)
        return [m[0].numpy() for m in masks]    # list of (height, width) True/False arrays


def mask_geometry(mask):
    """Centre (cx, cy), rotation angle and area of one mask."""
    ys, xs = np.nonzero(mask)                   # coordinates of all 'True' pixels
    if len(xs) == 0:
        return None
    cx, cy = int(xs.mean()), int(ys.mean())     # centre of mass of the object

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    (_, _), (w, h), angle = cv2.minAreaRect(biggest)   # smallest rotated rectangle around the object
    return {"centre": [cx, cy], "angle_deg": round(float(angle), 1),
            "area_px": int(mask.sum()), "contour": biggest}


def draw_masks(image, detections, masks):
    """Colour each mask, draw its outline, centre and angle. Returns a BGR image."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = img.copy()

    for i, mask in enumerate(masks):
        overlay[mask] = COLORS[i % len(COLORS)]          # paint the mask pixels
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)  # 45 % colour, 55 % original

    for i, (d, mask) in enumerate(zip(detections, masks)):
        color = COLORS[i % len(COLORS)]
        g = mask_geometry(mask)
        if g is None:
            continue
        cv2.drawContours(img, [g["contour"]], -1, color, 2)       # outline
        cx, cy = g["centre"]
        cv2.circle(img, (cx, cy), 6, (255, 255, 255), -1)         # centre point
        cv2.circle(img, (cx, cy), 6, color, 2)
        text = f'{d["label"]} {g["angle_deg"]}deg'
        cv2.putText(img, text, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img