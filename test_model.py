"""
Quick sanity-check — run this locally BEFORE pushing to Hugging Face.
Usage:
    python test_model.py                   # just loads the model
    python test_model.py path/to/image.jpg # runs inference on an image
"""
import sys
from ultralytics import YOLO

MODEL_PATH = "best.pt"

print(f"Loading model from '{MODEL_PATH}' …")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")
print(f"   Classes : {model.names}")
print(f"   Task    : {model.task}")

if len(sys.argv) > 1:
    img_path = sys.argv[1]
    print(f"\nRunning inference on '{img_path}' …")
    results = model(img_path, conf=0.35)
    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"   → {model.names[cls]}  conf={conf:.3f}  box={box.xyxy[0].tolist()}")
    print("✅ Inference complete!")
