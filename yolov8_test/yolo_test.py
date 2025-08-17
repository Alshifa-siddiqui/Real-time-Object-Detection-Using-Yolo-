from ultralytics import YOLO

# Load the pretrained YOLOv8 model (nano version – fast & small)
model = YOLO("yolov8n.pt")

# Run detection on a sample image from the internet
results = model("https://ultralytics.com/images/bus.jpg", show=True)

# Save the result as a local file
results[0].save(filename="yolo_output.jpg")
