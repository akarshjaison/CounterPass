import time
import threading
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.models.models import AnalysisJob, PlayerTrack
from app.services.detection import run_player_detection

from app.services.analytics_engine import run_tactical_analysis
from app.services.annotator import annotate_video

def start_analysis_job(job_id: int):
    """
    Spawns a background thread to process the analysis job.
    """
    thread = threading.Thread(
        target=_run_analysis_worker,
        args=(job_id,),
        daemon=True
    )
    thread.start()

def _ensure_player_tracks(db: Session, job_id: int):
    """
    Ensures that player track classifications are populated in the database.
    """
    from sqlalchemy import func
    from app.models.models import PlayerTrack, PlayerDetection

    existing_tracks = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id).first()
    if not existing_tracks:
        unique_tracks = db.query(
            PlayerDetection.track_id, 
            func.avg(PlayerDetection.confidence).label('avg_conf')
        ).filter(
            PlayerDetection.job_id == job_id,
            PlayerDetection.track_id.isnot(None),
            PlayerDetection.class_id == 0
        ).group_by(PlayerDetection.track_id).all()

        if not unique_tracks:
            raise ValueError("No players detected — check video quality/framing")

        for track_id, avg_conf in unique_tracks:
            # Fallback if detection.py failed to cluster teams
            team = "Team A" 
            db.add(PlayerTrack(job_id=job_id, track_id=track_id, team=team, confidence=float(avg_conf or 0.85)))
        db.commit()

def _run_analysis_worker(job_id: int):
    """
    Background worker that runs the full real CV analysis pipeline.
    """
    # Create a dedicated db session for the background thread
    db: Session = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            return

        job.status = "processing"
        job.progress = 0.0
        job.current_stage = "Initializing"
        db.commit()

        stages = [
            ("Extracting Frames", 0.0, 15.0),
            ("Detecting Players & Ball", 15.0, 35.0),
            ("Tracking Movement & Teams", 35.0, 55.0),
            ("Analyzing Possession & Lanes", 55.0, 75.0),
            ("Evaluating Passing Decisions", 75.0, 90.0),
        ]

        # Initialize Pitch Calibrator
        calibrator = None

        # Run stage processing
        for stage_name, start_pct, end_pct in stages:
            job.current_stage = stage_name
            job.progress = start_pct
            db.commit()
            
            if stage_name == "Detecting Players & Ball":
                video_path = job.video.path if job.video else None
                if video_path:
                    import cv2
                    from app.services.calibration import PitchCalibrator
                    from app.core.config import settings
                    import os
                    
                    # Calibrate pitch homography from first frame
                    pitch_model_path = os.path.join(settings.MODELS_DIR, "pose_field.pt")
                    calibrator = PitchCalibrator(pitch_model_path)
                    
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        calibrator.calibrate_from_frame(frame)
                        
                    run_player_detection(db, job_id, video_path, start_pct, end_pct)
            elif stage_name == "Tracking Movement & Teams":
                _ensure_player_tracks(db, job_id)
            elif stage_name == "Evaluating Passing Decisions":
                fps = job.video.fps or 30.0
                width = job.video.width or 1920
                height = job.video.height or 1080
                run_tactical_analysis(db, job_id, fps, width, height, calibrator)
                time.sleep(1.0)  # Small delay between stages

            job.progress = end_pct
            db.commit()

        # Finalize: generate annotated video
        time.sleep(1.0)
        annotate_video(db, job_id)

        job.status = "completed"
        job.progress = 100.0
        job.current_stage = "Finished"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        db.rollback()
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()
