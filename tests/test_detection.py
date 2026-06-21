"""Tests for the YOLOv8 detection entry point.

Unit tests for the source-routing/helpers run without heavy deps. The
integration test loads the real YOLOv8n model and runs detection on the
bundled bus.jpg; it is skipped if ultralytics/torch are unavailable or the
weights cannot be downloaded.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "yolov8_test"))

pytest.importorskip("cv2")
import main  # noqa: E402


# ---- unit tests (no model needed) ----

@pytest.mark.parametrize("src,expected", [
    ("bus.jpg", True),
    ("photo.PNG", True),
    ("clip.mp4", False),
    ("0", False),
    ("https://example.com/cat.jpg?w=10", True),
    ("https://example.com/stream", False),
])
def test_is_image_source(src, expected):
    assert main.is_image_source(src) is expected


def test_open_capture_bad_path_exits():
    with pytest.raises(SystemExit):
        main.open_capture("does_not_exist_12345.mp4")


def test_sample_image_present():
    assert os.path.exists(os.path.join(REPO_ROOT, "bus.jpg"))


# ---- integration test (downloads/loads YOLOv8n) ----

def test_detects_objects_in_bus_image():
    pytest.importorskip("ultralytics")
    pytest.importorskip("torch")
    from ultralytics import YOLO

    bus = os.path.join(REPO_ROOT, "bus.jpg")
    model = YOLO("yolov8n.pt")  # auto-downloads if absent
    results = model(bus, conf=0.25, verbose=False)
    n = len(results[0].boxes)
    assert n >= 1, "expected at least one detection in bus.jpg"
