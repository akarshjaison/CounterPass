import cv2
import os
from typing import Dict, Any

def get_video_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extracts video metadata (FPS, dimensions, frame count, duration) using OpenCV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found at {file_path}")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        # Fallback metadata if OpenCV cannot read the container format properly
        file_size = os.path.getsize(file_path)
        return {
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "duration": 0.0,
            "frame_count": 0,
            "error": "Could not open video file via OpenCV, using default structural placeholders."
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps > 0:
        duration = frame_count / fps
    else:
        fps = 30.0
        duration = 0.0

    cap.release()

    # Sanitization
    if width <= 0: width = 1920
    if height <= 0: height = 1080

    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "duration": float(duration),
        "frame_count": frame_count
    }
