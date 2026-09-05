import pytest
import numpy as np
from typing import cast
from app.services.calibration import PitchCalibrator, infer_team_attack_directions
from app.db.database import SessionLocal
from app.models.models import AnalysisJob, PlayerTrack, PlayerDetection, Video

def test_homography_transform_roundtrip():
    """
    Test that transform_point falls back to identity if uncalibrated, or uses transformer if available.
    """
    calibrator = PitchCalibrator()
    
    # Uncalibrated fallback
    tx0, ty0 = calibrator.transform_point(0.0, 0.0)
    assert tx0 == 0.0
    assert ty0 == 0.0

    # Mocking a transformer
    class MockTransformer:
        def transform_points(self, points):
            return points * 0.5
            
    calibrator.transformer = MockTransformer()
    tx1, ty1 = calibrator.transform_point(1920.0, 1080.0)
    assert tx1 == 960.0
    assert ty1 == 540.0

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

        # Cast job.id to int using typing.cast to satisfy the static type checker
        directions = infer_team_attack_directions(db, cast(int, job.id))
        assert directions["Team A"] == "right"
        assert directions["Team B"] == "left"
    finally:
        db.close()
