import pytest
import numpy as np
from app.services.calibration import compute_homography_matrix, transform_point, infer_team_attack_directions
from app.db.database import SessionLocal
from app.models.models import AnalysisJob, PlayerTrack, PlayerDetection, Video

def test_homography_transform_roundtrip():
    """
    Test that homography matrix transforms standard 4 corner points accurately.
    """
    src_points = [(0.0, 0.0), (1920.0, 0.0), (1920.0, 1080.0), (0.0, 1080.0)]
    H = compute_homography_matrix(src_points, img_width=1920, img_height=1080)

    # Top-left should map near (0, 0)
    tx0, ty0 = transform_point(0.0, 0.0, H)
    assert tx0 == pytest.approx(0.0, abs=1e-2)
    assert ty0 == pytest.approx(0.0, abs=1e-2)

    # Bottom-right should map near (105, 68)
    tx1, ty1 = transform_point(1920.0, 1080.0, H)
    assert tx1 == pytest.approx(105.0, abs=1e-2)
    assert ty1 == pytest.approx(68.0, abs=1e-2)

def test_attack_direction_inference():
    """
    Test infer_team_attack_directions assigns 'right' to the team positioned further left.
    """
    db = SessionLocal()
    try:
        video = Video(filename="test_calib.mp4", path="/tmp/test_calib.mp4")
        db.add(video)
        db.commit()

        job = AnalysisJob(video_id=video.id, status="processing")
        db.add(job)
        db.commit()

        db.add(PlayerTrack(job_id=job.id, track_id=1, team="Team A"))
        db.add(PlayerTrack(job_id=job.id, track_id=2, team="Team B"))

        # Team A positioned on the left side (x=200)
        db.add(PlayerDetection(
            job_id=job.id, frame_index=1, timestamp=5.0, track_id=1,
            x_min=190, y_min=100, x_max=210, y_max=140, center_x=200.0, center_y=120.0,
            confidence=0.9, class_id=0
        ))
        # Team B positioned on the right side (x=800)
        db.add(PlayerDetection(
            job_id=job.id, frame_index=1, timestamp=5.0, track_id=2,
            x_min=790, y_min=100, x_max=810, y_max=140, center_x=800.0, center_y=120.0,
            confidence=0.9, class_id=0
        ))
        db.commit()

        directions = infer_team_attack_directions(db, job.id)
        assert directions["Team A"] == "right"
        assert directions["Team B"] == "left"
    finally:
        db.close()
