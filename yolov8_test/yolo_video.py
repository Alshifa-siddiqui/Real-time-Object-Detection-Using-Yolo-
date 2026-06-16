"""Simple YOLOv8 video / webcam detection quickstart.

Usage:
    python yolov8_test/yolo_video.py                 # webcam (index 0)
    python yolov8_test/yolo_video.py path/to/video.mp4

For the full-featured CLI (images, --conf, --device, --save, ...), see main.py.
"""

import sys

import cv2
from ultralytics import YOLO

# Load the YOLOv8 nano model (fast & light). Auto-downloads on first run.
model = YOLO("yolov8n.pt")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Source: first CLI argument, or webcam index 0 by default.
source = sys.argv[1] if len(sys.argv) > 1 else "0"
cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)

if not cap.isOpened():
    sys.exit(f"Error: could not open video source '{source}'.")

# Get video properties, with safe fallbacks when the backend reports 0.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

# Create output writer
out = cv2.VideoWriter("output_detected.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Limit how many frames to process (optional)
max_frames = 300
frame_count = 0

while frame_count < max_frames:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO on this frame
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Draw results
    annotated_frame = results[0].plot()

    # Show live window
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Save the frame to output video
    out.write(annotated_frame)
    frame_count += 1

    # Quit on 'q' key...
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # ...or when the window is closed via the X button.
    if cv2.getWindowProperty("YOLOv8 Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Processed {frame_count} frames and saved to 'output_detected.mp4'")
