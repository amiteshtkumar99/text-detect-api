# api.py - web service for text-prompted object detection

import io

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from detector import TextDetector, draw_detections

app = FastAPI(title="Text-Prompted Object Detection API")

# The model is loaded ONCE when the server starts, not for every request
detector = TextDetector(fp16=True)  # use fp16 on GPU for speed and lower memory usage


def read_image(upload: UploadFile) -> Image.Image:
    """Turn the uploaded file into a PIL image, or answer with an error."""
    try:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image file")


@app.get("/health")
def health():
    """Quick check: is the server running, and on which device?"""
    return {"status": "ok", "device": detector.device}


@app.post("/detect")
def detect(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    threshold: float = Form(0.4),
):
    """Returns the detections as JSON."""
    pil_image = read_image(image)
    detections = detector.detect(pil_image, prompt, box_threshold=threshold)
    return {"prompt": prompt, "count": len(detections), "detections": detections, "timings_ms": detector.last_timings}


@app.post("/detect/image")
def detect_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    threshold: float = Form(0.4),
):
    """Returns the image with boxes drawn on it."""
    pil_image = read_image(image)
    detections = detector.detect(pil_image, prompt, box_threshold=threshold)
    drawn = draw_detections(pil_image, detections)
    ok, buffer = cv2.imencode(".jpg", drawn)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")