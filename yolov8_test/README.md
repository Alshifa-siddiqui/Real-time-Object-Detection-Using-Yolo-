# Task 3 — Real-time Object Detection with YOLOv8

This project runs YOLOv8 in real-time using a webcam, video, or image.
Objects are detected and displayed with bounding boxes + labels.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## Run

* Webcam:

```bash
python main.py
```

* Video file:

```bash
python main.py --source sample.mp4
```

* Image:

```bash
python main.py --source bus.jpg
```

Press `q` to quit webcam mode.

## Notes

* Uses YOLOv8n (nano, fast).
* Model file (`yolov8n.pt`) auto-downloads on first run.
* Results are displayed in a window. For images, press any key to close the window.

## Folder Structure

```
object-detection-project/
├─ .venv/              ← virtual environment (ignored by Git)
├─ main.py             ← main detection script
├─ requirements.txt    ← dependencies
├─ .gitignore          ← ignore rules
└─ README.md           ← this file
```

## Author

Prepared as **Task 3 submission** for internship project.
