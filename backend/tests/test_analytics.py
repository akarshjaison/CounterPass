import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.db.database import SessionLocal
from app.models.models import PlayerDetection, PlayerTrack, PassEvent, PassingOption, MissedOpportunity
from app.services.analysis import _run_analysis_worker

client = TestClient(app)

def test_real_cv_analytics_pipeline():
    """
    Integration test verifying that the real CV analytics pipeline processes detections,
    reconstructs plays, estimates possession, detects pass events, evaluates passing options,
    determines missed opportunities, compiles metrics, and generates an annotated output video.
    """
    # 1. Upload video
    files = {"file": ("test_real_run.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")}
    upload_res = client.post(f"{settings.API_V1_STR}/videos/upload", files=files)
    assert upload_res.status_code == 200
    video_data = upload_res.json()
    video_id = video_data["id"]
    video_path = video_data["path"]

    # 2. Create Job in "real" mode
    job_payload = {"video_id": video_id, "mode": "real"}
    job_res = client.post(f"{settings.API_V1_STR}/analysis/start/{video_id}", json=job_payload)
    assert job_res.status_code == 200
    job_data = job_res.json()
    job_id = job_data["id"]

    # 3. Run background worker synchronously
    # This will trigger: run_player_detection -> _ensure_player_tracks -> run_tactical_analysis -> annotate_video
    _run_analysis_worker(job_id, mode="real")

    # 4. Verify job status is completed
    status_res = client.get(f"{settings.API_V1_STR}/analysis/{job_id}/status")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "completed"

    # 5. Verify database metrics are compiled
    results_res = client.get(f"{settings.API_V1_STR}/analysis/{job_id}/results")
    assert results_res.status_code == 200
    results_data = results_res.json()
    assert results_data["total_passes"] > 0
    assert results_data["completion_rate"] >= 0.0
    assert results_data["counterpass_score"] >= 0.0
    assert "avg_option_score" in results_data

    # 6. Verify pass events are populated with dynamic pitch coordinates
    events_res = client.get(f"{settings.API_V1_STR}/analysis/{job_id}/events")
    assert events_res.status_code == 200
    events_data = events_res.json()
    assert len(events_data) > 0
    
    first_event = events_data[0]
    assert "passer_track_id" in first_event
    assert "receiver_track_id" in first_event
    # Verify coordinates are mapped as percentages (0-100)
    assert first_event["passer_x"] is not None
    assert 0.0 <= first_event["passer_x"] <= 100.0
    assert first_event["receiver_x"] is not None
    assert 0.0 <= first_event["receiver_x"] <= 100.0
    assert len(first_event["opponents"]) > 0
    assert first_event["opponents"][0]["x"] >= 0.0
    
    # Check that option coordinates are populated
    assert len(first_event["options"]) > 0
    assert first_event["options"][0]["x"] is not None
    assert 0.0 <= first_event["options"][0]["x"] <= 100.0

    # 7. Verify annotated output video is generated on disk
    annotated_filename = f"job_{job_id}_annotated.mp4"
    annotated_path = os.path.join(settings.OUTPUT_DIR, annotated_filename)
    assert os.path.exists(annotated_path)

    # 8. Verify the video endpoint streams the annotated version
    video_res = client.get(f"{settings.API_V1_STR}/analysis/{job_id}/video")
    assert video_res.status_code == 200
    assert video_res.headers.get("content-type") == "video/mp4"

    # Cleanup
    if video_path and os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(annotated_path):
        os.remove(annotated_path)
