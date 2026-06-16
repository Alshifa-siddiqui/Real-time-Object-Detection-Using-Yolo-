"""Minimal YOLOv8 single-image detection quickstart."""

import os

from ultralytics import YOLO

# Load the pretrained YOLOv8 model (nano version - fast & small).
# Auto-downloads the weights on first run.
model = YOLO("yolov8n.pt")

# Use the bundled sample image if present, else fall back to the online sample.
SOURCE = "bus.jpg" if os.path.exists("bus.jpg") else "https://ultralytics.com/images/bus.jpg"

# Run detection (show=True opens a preview window).
results = model(SOURCE, show=True)

# Save the result as a local file.
results[0].save(filename="yolo_output.jpg")
print("Saved annotated image to 'yolo_output.jpg'")
