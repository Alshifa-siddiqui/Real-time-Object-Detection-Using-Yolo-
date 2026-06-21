# Real-time Object Detection using YOLOv8

Internship **Task 3** — real-time object detection with a pre-trained YOLOv8 model.  
Supports **webcam**, **video files**, and **single images**. Shows **bounding boxes + labels + confidence** in real time.

## 1) Project Structure

```
.
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ run.bat                     # Windows convenience launcher
├─ bus.jpg                     # bundled sample image
├─ yolo_output.jpg             # sample detection output
├─ .github/workflows/ci.yml    # CI: install + tests
├─ tests/
│  └─ test_detection.py        # pytest suite (unit + integration)
└─ yolov8_test/
   ├─ main.py                  # full CLI: webcam / video / image (recommended)
   ├─ yolo_test.py             # minimal single-image quickstart
   └─ yolo_video.py            # minimal video / webcam quickstart
```

> Code lives in `yolov8_test/`, so use the commands below as-is. Model weights
> (`yolov8n.pt`) download automatically on first run and are git-ignored.

## 2) Setup

**Python:** 3.10–3.11 recommended

```bash
# create & activate venv (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

# or on macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt
```

If `torch` didn’t install automatically, run:

```bash
# CPU-only (safe default)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For CUDA users, grab the right command from PyTorch’s site.

## 3) Run

### A) Webcam (default camera index 0)
```bash
python yolov8_test/main.py --source 0 --model yolov8n.pt
```

### B) Video file
```bash
python yolov8_test/main.py --source path/to/video.mp4 --model yolov8n.pt
```

### C) Single image
```bash
python yolov8_test/main.py --source path/to/image.jpg --model yolov8n.pt --save
```

> First run auto-downloads `yolov8n.pt` if it’s not present.

## 4) Options you can tweak

- `--model`: `yolov8n.pt` (fast) | `yolov8s.pt` (better) | `yolov8m.pt` (even better, slower)
- `--conf`: confidence threshold (e.g. `--conf 0.25`)
- `--save`: save annotated frames / outputs
- `--show`: force a display window (if you disabled it in code)
- `--device`: pick `cpu` or a cuda id, e.g. `--device 0`

Example:
```bash
python yolov8_test/main.py --source 0 --model yolov8s.pt --conf 0.35
```

## 5) Results (proof it works)

Verified single-image detection on the bundled `bus.jpg` (CPU, yolov8n):

```
$ python yolov8_test/main.py --source bus.jpg --save --no-show
image 1/1 bus.jpg: 640x480 4 persons, 1 bus, 1 stop sign, 108.7ms
Detected 6 object(s).
Saved annotated image to 'yolo_output.jpg'.
```

Annotated output is saved as `yolo_output.jpg`.

**Observed FPS (yolov8n, CPU):** ~12–18 FPS @ 288×640 on a laptop.
**Per-image latency (yolov8n, CPU):** ~100–110 ms @ 640×480.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Unit tests cover source routing and capture error handling; the integration
test loads YOLOv8n and asserts detections on `bus.jpg` (skipped automatically
if `ultralytics`/`torch` are unavailable). CI runs the suite on every push/PR.

## 6) Known Issues / Tips

- **Webcam not opening**: try `--source 1` or close other apps using the camera.
- **macOS screen permission**: grant Terminal/VS Code camera permission.
- **Slow FPS on CPU**: switch to `yolov8n.pt`, reduce frame size, or use GPU (`--device 0`).
- **Torch install weirdness**: use the CPU wheel above or install the exact CUDA wheel for your GPU.

## 7) Why this meets the requirements

- Uses **pre-trained YOLOv8** model.
- Handles **webcam, video, image** inputs.
- Draws **boxes + labels + confidence** in real time.
- Includes **reproducible setup**, **clear run commands**, and **evidence (screens/GIF)**.
- Notes on **trade-offs** between `n/s/m` models.

## 8) License

Released under the [MIT License](LICENSE).

## 9) Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV for video I/O
