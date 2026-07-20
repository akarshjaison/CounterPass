import os
# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.services.analysis import _run_analysis_worker
from app.db.database import SessionLocal

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
    """
    # 1. Upload video
    files = {"file": ("test_run.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")}
    upload_res = client.post(f"{settings.API_V1_STR}/videos/upload", files=files)
    assert upload_res.status_code == 200
    video_data = upload_res.json()
    video_id = video_data["id"]
    video_path = video_data["path"]

    # 2. Create Job
    job_payload = {"video_id": video_id, "mode": "demo"}
    job_res = client.post(f"{settings.API_V1_STR}/analysis/start/{video_id}", json=job_payload)
    assert job_res.status_code == 200
    job_data = job_res.json()
    job_id = job_data["id"]

    # 3. Execute background worker synchronously for deterministic testing
    db = SessionLocal()
    try:
        _run_analysis_worker(job_id, mode="demo")
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
    
    # We downsample detections, so there should be a reasonable number
    assert len(detections) > 0
    
    # Validate structure of first detection
    first_det = detections[0]
    assert "frame_index" in first_det
    assert "timestamp" in first_det
    assert "x_min" in first_det
    assert "center_x" in first_det
    assert "confidence" in first_det
    assert "class_id" in first_det
    assert first_det["job_id"] == job_id

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
    # Check that position was estimated with velocity
    # velocity in f2 was: center(f2) - center(f1) = (22, 22) - (20, 20) = (2, 2)
    # with damping: 0.6 * 2.0 + 0.4 * 0.0 = 1.2
    # in f3 prediction: center + velocity = 22 + 1.2 = 23.2
    assert tracker.tracks[track_id]['center'][0] == pytest.approx(23.2)
    
    # Frame 4: Object reappears near predicted location
    dets_f4 = [
        {'box': [14.0, 14.0, 34.0, 34.0], 'confidence': 0.88}
    ]
    tracks_f4 = tracker.update(dets_f4)
    assert len(tracks_f4) == 1
    assert tracks_f4[0]['id'] == track_id
    assert tracks_f4[0]['state'] == 'tracked'
