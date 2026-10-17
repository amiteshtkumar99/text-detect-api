# detector.py - reusable text-prompted object detector
import time

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


class TextDetector:
    """Loads Grounding DINO once and detects objects from a text prompt."""

    def __init__(self, model_id=MODEL_ID, fp16=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = fp16 and self.device == "cuda"
        print(f"Loading model on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()   # tell the model we only predict, no training
        self.last_timimgs = {}
        
    def _sync(self):
        """Wait until the GPU has really finished (otherwise timings lie)."""
        if self.device == "cuda":
            torch.cuda.synchronize()

    def detect(self, image, prompt, box_threshold=0.4, text_threshold=0.3):
        """image: a PIL image (RGB). Returns a list of detections."""
        t0 = time.perf_counter()
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        self._sync()
        t1 = time.perf_counter()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            outputs = self.model(**inputs)
        self._sync()
        t2 = time.perf_counter()

        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        t3 = time.perf_counter()

        self.last_timings = {
            "preprocess_ms": round((t1 - t0) * 1000, 1),
            "model_ms": round((t2 - t1) * 1000, 1),
            "postprocess_ms": round((t3 - t2) * 1000, 1),
        }

        detections = []
        for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            detections.append({
                "label": label,
                "score": round(float(score), 3),
                "box": [x1, y1, x2, y2],
            })
        return detections


def draw_detections(image, detections):
    """Draw boxes and labels. Returns an OpenCV image (BGR) ready to save."""
    # PIL uses RGB colour order, OpenCV uses BGR -> convert
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        text = f'{d["label"]} {d["score"]:.2f}'

        # 1. the box (green, 2 px thick)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 2. a filled label background above the box
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1 - th - 8, 0)            # stay inside the image
        cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + th + 8), (0, 255, 0), -1)

        # 3. the label text (black)
        cv2.putText(img, text, (x1 + 3, top + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img