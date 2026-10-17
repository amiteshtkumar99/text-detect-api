# client.py - talks to the detection API like a robot program would

import argparse
import statistics
import time

import requests

API_URL = "http://localhost:8000"

parser = argparse.ArgumentParser(description="Client for the detection API")
parser.add_argument("image", help="path to the image to send")
parser.add_argument("prompt", help='e.g. "a cat. a remote control."')
parser.add_argument("--threshold", type=float, default=0.4)
parser.add_argument("--runs", type=int, default=10, help="how many times to send (for speed test)")
args = parser.parse_args()

# 1. Is the server alive?
health = requests.get(f"{API_URL}/health", timeout=5)
print("Server:", health.json())

# 2. Send the same image several times and measure the time
times_ms = []
for i in range(args.runs):
    with open(args.image, "rb") as f:                     # "rb" = read bytes
        start = time.time()
        response = requests.post(
            f"{API_URL}/detect",
            files={"image": f},
            data={"prompt": args.prompt, "threshold": args.threshold},
            timeout=60,
        )
        times_ms.append((time.time() - start) * 1000)

    response.raise_for_status()                          # stop if the server answered with an error

result = response.json()                                 # JSON text -> Python dictionary
print("Server timings:", result.get("timings_ms"))

# 3. Use the answer: like a robot choosing what to pick
print(f"\nFound {result['count']} objects:")
for d in result["detections"]:
    x1, y1, x2, y2 = d["box"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2               # centre of the box
    print(f"  {d['label']:18s} score={d['score']:.2f}  centre=({cx}, {cy})")

if result["detections"]:
    best = max(result["detections"], key=lambda d: d["score"])
    print(f"\nBest target to pick: {best['label']} (score {best['score']:.2f})")

# 4. Speed report (skip the first run: GPU warm-up)
warm = times_ms[1:] if len(times_ms) > 1 else times_ms
print(f"\nFirst request : {times_ms[0]:.0f} ms (includes warm-up)")
print(f"Average after : {statistics.mean(warm):.0f} ms")
print(f"Fastest       : {min(warm):.0f} ms")
print(f"≈ {1000 / statistics.mean(warm):.1f} images per second")

# 5. Also get the drawn image back and save it
with open(args.image, "rb") as f:
    img_response = requests.post(
        f"{API_URL}/detect/image",
        files={"image": f},
        data={"prompt": args.prompt, "threshold": args.threshold},
        timeout=60,
    )
img_response.raise_for_status()
with open("outputs/client_result.jpg", "wb") as out:      # "wb" = write bytes
    out.write(img_response.content)
print("Saved drawn image: outputs/client_result.jpg")