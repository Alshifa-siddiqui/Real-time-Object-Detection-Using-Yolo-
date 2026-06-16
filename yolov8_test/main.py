"""Real-time object detection with YOLOv8.

Unified entry point supporting three input types:

  * Webcam   -> --source 0   (camera index, default)
  * Video    -> --source path/to/video.mp4
  * Image    -> --source path/to/image.jpg  (local path or http(s) URL)

Draws bounding boxes + labels + confidence. Press 'q' to quit a live window.

Examples
--------
    python yolov8_test/main.py --source 0 --model yolov8n.pt
    python yolov8_test/main.py --source video.mp4 --conf 0.35 --save
    python yolov8_test/main.py --source bus.jpg --save
"""

import argparse
import os
import sys

import cv2
from ultralytics import YOLO

# File extensions we treat as still images rather than video streams.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time object detection using YOLOv8."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (e.g. 0), a video/image path, or an image URL. Default: 0 (webcam).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Model weights. yolov8n.pt (fast) | yolov8s.pt | yolov8m.pt. Auto-downloads if missing.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (0-1). Default: 0.25."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'cpu' or a CUDA id like '0'. Default: auto.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames for video/webcam (0 = no limit). Default: 0.",
    )
    parser.add_argument(
        "--output",
        default="output_detected.mp4",
        help="Output video path when --save is used on video/webcam.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save annotated output (video file or image)."
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Disable the live display window (useful on headless machines).",
    )
    parser.set_defaults(show=True)
    return parser.parse_args()


def is_image_source(source: str) -> bool:
    """Return True if the source string points to a still image (local or URL)."""
    lowered = source.lower().split("?")[0]  # drop URL query strings
    return os.path.splitext(lowered)[1] in IMAGE_EXTS


def run_image(model: YOLO, args) -> None:
    """Run detection on a single image and optionally save it."""
    results = model(args.source, conf=args.conf, device=args.device, show=args.show)
    result = results[0]
    print(f"Detected {len(result.boxes)} object(s).")
    if args.save:
        out_path = "yolo_output.jpg"
        result.save(filename=out_path)
        print(f"Saved annotated image to '{out_path}'.")
    if args.show:
        # Keep the window open until a key is pressed (Ultralytics show=True does
        # not block for images).
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a webcam index or a video file, returning a validated capture."""
    # A bare integer string means a webcam index; otherwise treat as a path.
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        hint = (
            "Is another app using the camera? Try a different index (e.g. 1)."
            if source.isdigit()
            else "Check that the file exists and the path is correct."
        )
        sys.exit(f"Error: could not open video source '{source}'. {hint}")
    return cap


def run_video(model: YOLO, args) -> None:
    """Run detection frame-by-frame on a webcam or video file."""
    cap = open_capture(args.source)

    # Probe properties, falling back to safe defaults if the backend reports 0
    # (common for some webcams) so the VideoWriter is always valid.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    writer = None
    if args.save:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            sys.exit(f"Error: could not open VideoWriter for '{args.output}'.")

    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, conf=args.conf, device=args.device, verbose=False)
            annotated = results[0].plot()

            if args.show:
                cv2.imshow("YOLOv8 Detection", annotated)
            if writer is not None:
                writer.write(annotated)

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
            if args.show and (cv2.waitKey(1) & 0xFF == ord("q")):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Done. Processed {frame_count} frame(s).")
    if args.save:
        print(f"Saved annotated video to '{args.output}'.")


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    if is_image_source(args.source):
        run_image(model, args)
    else:
        run_video(model, args)


if __name__ == "__main__":
    main()
