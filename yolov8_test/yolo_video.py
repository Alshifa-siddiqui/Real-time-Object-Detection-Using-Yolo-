from ultralytics import YOLO
import cv2

# Load the YOLOv8 nano model (fast & light)
model = YOLO("yolov8n.pt")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Load your video
video_path = "C:\\Users\\Admin\\OneDrive\\Desktop\\video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# Create output writer
out = cv2.VideoWriter("output_detected.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Limit how many frames to process (optional)
max_frames = 300
frame_count = 0

while cap.isOpened() and frame_count < max_frames:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO on this frame
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    # Draw results
    annotated_frame = results[0].plot()

    # Show live window
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Save the frame to output video
    out.write(annotated_frame)
    frame_count += 1

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Processed {frame_count} frames and saved to 'output_detected.mp4'")
