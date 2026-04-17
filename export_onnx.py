"""
Export YOLO pose model to ONNX for faster CPU inference.

Run once:
    python export_onnx.py

Then use the exported model:
    python main.py --model yolo26n-pose.onnx --no-buzzer --no-servo
"""

from ultralytics import YOLO

MODEL_IN  = "yolo26n-pose.pt"
IMGSZ     = 256          # match --imgsz you plan to use at runtime

print(f"[EXPORT] Loading {MODEL_IN} ...")
model = YOLO(MODEL_IN)

print(f"[EXPORT] Exporting to ONNX at imgsz={IMGSZ} ...")
path = model.export(format="onnx", imgsz=IMGSZ, simplify=True)

print(f"[EXPORT] Done → {path}")
print()
print("Run with:")
print(f"  python main.py --model yolo26n-pose.onnx --no-buzzer --no-servo")
