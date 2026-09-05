import os
# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.core.config import settings
from app.services.analysis import _run_analysis_worker
from app.db.database import SessionLocal
from app.models.models import PlayerDetection

client = TestClient(app)

def test_health_check():
    """
    Test the health check endpoint returns 200 and 'healthy' status.
    """
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_upload_invalid_file_type():
    """
    Test that uploading an unsupported file type (like .txt) returns 400.
    """
    # Create a dummy text file
    files = {"file": ("test.txt", b"Hello football analysis!", "text/plain")}
    response = client.post(f"{settings.API_V1_STR}/videos/upload", files=files)
    assert response.status_code == 400
    assert "Invalid video format" in response.json()["detail"]

def test_upload_valid_mock_video():
    """
    Test that uploading a file with a valid extension returns 200 and registers the video.
    """
    # We write a mock binary payload resembling a video file
    files = {"file": ("test_match.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")}
    response = client.post(f"{settings.API_V1_STR}/videos/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_match.mp4"
    assert data["fps"] == 30.0  # Fallback FPS
    assert data["width"] == 1920  # Fallback width
    assert data["height"] == 1080  # Fallback height
    assert "id" in data
    assert "path" in data

    # Cleanup uploaded file from disk if it was created
    uploaded_path = data.get("path")
    if uploaded_path and os.path.exists(uploaded_path):
        os.remove(uploaded_path)

def test_analysis_pipeline_and_detections():
    """
    Test uploading a video, creating a job, running the analysis, 
    and verifying that player detections are correctly populated and queried.
    Detection is mocked so the test runs without YOLO weights or a real video.
    """
    # 1. Upload video
    files = {"file": ("test_run.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")}
    upload_res = client.post(f"{settings.API_V1_STR}/videos/upload", files=files)
    assert upload_res.status_code == 200
    video_data = upload_res.json()
    video_id = video_data["id"]
    video_path = video_data["path"]

    # 2. Create Job
    job_payload = {"video_id": video_id}
    job_res = client.post(f"{settings.API_V1_STR}/analysis/start/{video_id}", json=job_payload)
    assert job_res.status_code == 200
    job_data = job_res.json()
    job_id = job_data["id"]

    # 3. Execute background worker synchronously (mock detection to avoid ONNX download)
    def _mock_detection(db, job_id, video_path):
        for frame_idx in range(5):
            ts = frame_idx / 30.0
            db.add(PlayerDetection(
                job_id=job_id, frame_index=frame_idx, timestamp=ts,
                track_id=1, x_min=100.0, y_min=100.0, x_max=120.0, y_max=140.0,
                center_x=110.0, center_y=120.0, confidence=0.9, class_id=0
            ))
        db.commit()

    db = SessionLocal()
    try:
        with patch("app.services.analysis.run_player_detection", side_effect=_mock_detection):
            _run_analysis_worker(job_id)
    finally:
        db.close()

    # 4. Check status is completed
    status_res = client.get(f"{settings.API_V1_STR}/analysis/{job_id}/status")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "completed"

    # 5. Query Detections endpoint
    det_res = client.get(f"{settings.API_V1_STR}/analysis/{job_id}/detections")
    assert det_res.status_code == 200
    detections = det_res.json()
    
    # We expect populated detections list
    assert isinstance(detections, list)
    assert len(detections) > 0

    # Cleanup video from disk
    if video_path and os.path.exists(video_path):
        os.remove(video_path)

def test_tracker_logic():
    """
    Test that SimpleTracker correctly associates detections across frames
    and maintains persistent track IDs.
    """
    from app.services.tracker import SimpleTracker
    
    tracker = SimpleTracker(max_lost_frames=3)
    
    # Frame 1: Initial detection
    dets_f1 = [
        {'box': [10.0, 10.0, 30.0, 30.0], 'confidence': 0.9}
    ]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 1
    track_id = tracks_f1[0]['id']
    assert tracks_f1[0]['state'] == 'tracked'
    
    # Frame 2: Same object moved slightly (high IoU)
    dets_f2 = [
        {'box': [12.0, 12.0, 32.0, 32.0], 'confidence': 0.85}
    ]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 1
    assert tracks_f2[0]['id'] == track_id
    assert tracks_f2[0]['state'] == 'tracked'
    
    # Frame 3: Object missing (lost state)
    tracks_f3 = tracker.update([])
    # tracker.update returns only currently active (visible) tracks, but keeps lost tracks internally
    assert len(tracks_f3) == 0
    assert track_id in tracker.tracks
    assert tracker.tracks[track_id]['state'] == 'lost'
    
    # Frame 4: Object reappears near predicted location
    dets_f4 = [
        {'box': [14.0, 14.0, 34.0, 34.0], 'confidence': 0.88}
    ]
    tracks_f4 = tracker.update(dets_f4)
    assert len(tracks_f4) == 1
    assert tracks_f4[0]['id'] == track_id
    assert tracks_f4[0]['state'] == 'tracked'
