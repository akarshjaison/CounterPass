import time
import threading
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.models.models import Video, AnalysisJob, PlayerTrack, PassEvent, PassingOption, MissedOpportunity
from app.services.detection import run_player_detection

from app.services.analytics_engine import run_tactical_analysis
from app.services.annotator import annotate_video

def start_analysis_job(job_id: int, mode: str = "real"):
    """
    Spawns a background thread to process the analysis job.
    """
    thread = threading.Thread(
        target=_run_analysis_worker,
        args=(job_id, mode),
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
            unique_tracks = [(i, 0.85) for i in range(1, 6)] + [(i, 0.80) for i in range(12, 17)]

        for track_id, avg_conf in unique_tracks:
            if track_id == 99:
                team = "Referee"
            elif track_id % 2 == 0:
                team = "Team A"
            else:
                team = "Team B"
            db.add(PlayerTrack(job_id=job_id, track_id=track_id, team=team, confidence=float(avg_conf or 0.85)))
        db.commit()

def _run_analysis_worker(job_id: int, mode: str):
    """
    Background worker that runs the processing simulation or actual analysis.
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
            ("Extracting Frames", 15.0),
            ("Detecting Players & Ball", 35.0),
            ("Tracking Movement & Teams", 55.0),
            ("Analyzing Possession & Lanes", 75.0),
            ("Evaluating Passing Decisions", 90.0),
        ]

        # Run stage processing
        for stage_name, progress_pct in stages:
            job.current_stage = stage_name
            job.progress = progress_pct
            db.commit()
            
            if stage_name == "Detecting Players & Ball":
                video_path = job.video.path if job.video else None
                if video_path:
                    run_player_detection(db, job_id, video_path, mode)
            elif stage_name == "Tracking Movement & Teams":
                if mode == "real":
                    _ensure_player_tracks(db, job_id)
                else:
                    time.sleep(1.0)
            elif stage_name == "Evaluating Passing Decisions":
                if mode == "real":
                    fps = job.video.fps or 30.0
                    width = job.video.width or 1920
                    height = job.video.height or 1080
                    run_tactical_analysis(db, job_id, fps, width, height)
                else:
                    time.sleep(1.0)
            else:
                time.sleep(1.0)  # Delay between simulated stages

        # Finalize processing
        time.sleep(1.0)
        
        # Populate Mock Data for Demo or Run Annotator for Real
        if mode == "demo":
            _populate_demo_results(db, job_id)
        else:
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

def _populate_demo_results(db: Session, job_id: int):
    """
    Populates SQLite database with highly realistic analytics for demo mode.
    """
    # 1. Add Player Tracks (Team A, Team B, Referee)
    # Team A: IDs 1-11
    # Team B: IDs 12-22
    # Referee: ID 99
    tracks = []
    for i in range(1, 12):
        tracks.append(PlayerTrack(job_id=job_id, track_id=i, team="Team A", confidence=0.95))
    for i in range(12, 23):
        tracks.append(PlayerTrack(job_id=job_id, track_id=i, team="Team B", confidence=0.92))
    tracks.append(PlayerTrack(job_id=job_id, track_id=99, team="Referee", confidence=0.89))
    db.add_all(tracks)
    db.commit()

    # 2. Add Pass Events and Passing Options
    # Pass 1: Successful short pass
    p1 = PassEvent(
        job_id=job_id, passer_track_id=8, receiver_track_id=10, 
        timestamp=4.5, outcome="completed", confidence=0.96
    )
    db.add(p1)
    db.commit() # Commit to get pass ID

    db.add_all([
        PassingOption(
            pass_event_id=p1.id, candidate_track_id=10, source="observed", 
            score=0.88, confidence=0.96, explanation="Selected Option. Clear passing corridor with moderate space to turn."
        ),
        PassingOption(
            pass_event_id=p1.id, candidate_track_id=7, source="observed", 
            score=0.82, confidence=0.94, explanation="Teammate open on right flank. Safe lateral movement option."
        ),
        PassingOption(
            pass_event_id=p1.id, candidate_track_id=9, source="temporally_inferred", 
            score=0.65, confidence=0.72, explanation="Inferred option behind central midfielder. Track temporarily lost due to referee occlusion."
        ),
        PassingOption(
            pass_event_id=p1.id, candidate_track_id=11, source="observed", 
            score=0.45, confidence=0.90, explanation="High-pressure forward option. Lane partially blocked by defender 14."
        )
    ])

    # Pass 2: Successful long pass (Team B)
    p2 = PassEvent(
        job_id=job_id, passer_track_id=14, receiver_track_id=17, 
        timestamp=12.2, outcome="completed", confidence=0.90
    )
    db.add(p2)
    db.commit()

    db.add_all([
        PassingOption(
            pass_event_id=p2.id, candidate_track_id=17, source="observed", 
            score=0.75, confidence=0.90, explanation="Selected Option. Forward run behind Team A defensive line. Risk of offside but high progression value."
        ),
        PassingOption(
            pass_event_id=p2.id, candidate_track_id=16, source="observed", 
            score=0.72, confidence=0.91, explanation="Teammate in lateral space. Safe support option with zero pressure."
        ),
        PassingOption(
            pass_event_id=p2.id, candidate_track_id=18, source="observed", 
            score=0.58, confidence=0.88, explanation="Central passing lane blocked by opponent 5. High interception risk."
        )
    ])

    # Pass 3: Intercepted Pass (Team A)
    p3 = PassEvent(
        job_id=job_id, passer_track_id=10, receiver_track_id=7, 
        timestamp=19.8, outcome="intercepted", confidence=0.92
    )
    db.add(p3)
    db.commit()

    db.add_all([
        PassingOption(
            pass_event_id=p3.id, candidate_track_id=7, source="observed", 
            score=0.42, confidence=0.92, explanation="Selected Option. Narrow passing lane. Defender 15 was in active interception path and cut off the ball."
        ),
        PassingOption(
            pass_event_id=p3.id, candidate_track_id=11, source="observed", 
            score=0.89, confidence=0.95, explanation="Missed Opportunity. Open teammate on the left wing. Lane completely clear of defenders with 12m of space."
        ),
        PassingOption(
            pass_event_id=p3.id, candidate_track_id=8, source="observed", 
            score=0.61, confidence=0.93, explanation="Backward pass option to retain possession under moderate pressure."
        )
    ])

    # Pass 4: Completed pass under pressure (Team B)
    p4 = PassEvent(
        job_id=job_id, passer_track_id=18, receiver_track_id=20, 
        timestamp=28.5, outcome="completed", confidence=0.93
    )
    db.add(p4)
    db.commit()

    db.add_all([
        PassingOption(
            pass_event_id=p4.id, candidate_track_id=20, source="observed", 
            score=0.78, confidence=0.93, explanation="Selected Option. Received under pressure, but player successfully shielded the ball."
        ),
        PassingOption(
            pass_event_id=p4.id, candidate_track_id=22, source="temporally_inferred", 
            score=0.81, confidence=0.68, explanation="Teammate making deep run. Occluded behind opponent 4, but trajectory suggests lane was highly valuable."
        )
    ])

    # Pass 5: Unsuccessful pass out of bounds
    p5 = PassEvent(
        job_id=job_id, passer_track_id=7, receiver_track_id=9, 
        timestamp=37.1, outcome="unsuccessful", confidence=0.85
    )
    db.add(p5)
    db.commit()

    db.add_all([
        PassingOption(
            pass_event_id=p5.id, candidate_track_id=9, source="observed", 
            score=0.38, confidence=0.85, explanation="Selected Option. Direct pass down the sideline was overhit, rolling out of bounds before receiver could reach it."
        ),
        PassingOption(
            pass_event_id=p5.id, candidate_track_id=8, source="observed", 
            score=0.76, confidence=0.92, explanation="Simple square pass to pivot the play. Safe option with low interception probability."
        )
    ])
    db.commit()

    # 3. Add Missed Opportunities
    db.add_all([
        MissedOpportunity(
            job_id=job_id, timestamp=19.8, carrier_track_id=10, recommended_track_id=11,
            score=0.89, confidence=0.95,
            explanation="Player 11 was completely unmarked in the left channel with 8m of space. Passer chose a high-risk lane to Player 7, leading to an interception by Player 15."
        ),
        MissedOpportunity(
            job_id=job_id, timestamp=35.0, carrier_track_id=9, recommended_track_id=8,
            score=0.84, confidence=0.88,
            explanation="Player 8 opened a clean forward passing lane in the center channel for 1.2 seconds, but Player 9 delayed the pass and got tackled by Player 13."
        )
    ])
    db.commit()

def _populate_stub_real_results(db: Session, job_id: int):
    """
    Populates SQLite database with basic stub metrics for Real CV mode in Phase 1 & 2 & 3 & 5.
    """
    from sqlalchemy import func
    from app.models.models import PlayerTrack, PlayerDetection

    # Check if PlayerTrack records already exist for this job (populated by YOLO run)
    existing_tracks = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id).first()
    if existing_tracks:
        # PlayerTracks already exist, just retrieve the unique track IDs for mock pass event
        unique_tracks_query = db.query(
            PlayerDetection.track_id
        ).filter(
            PlayerDetection.job_id == job_id,
            PlayerDetection.track_id.isnot(None),
            PlayerDetection.class_id == 0
        ).group_by(PlayerDetection.track_id).all()
        unique_tracks = [(t[0], 0.85) for t in unique_tracks_query]
    else:
        # Find all unique track IDs from PlayerDetection for this job (excluding ball)
        unique_tracks = db.query(
            PlayerDetection.track_id, 
            func.avg(PlayerDetection.confidence).label('avg_conf')
        ).filter(
            PlayerDetection.job_id == job_id,
            PlayerDetection.track_id.isnot(None),
            PlayerDetection.class_id == 0
        ).group_by(PlayerDetection.track_id).all()

        # Fallback to default lists if no detections were found (e.g. empty video)
        if not unique_tracks:
            unique_tracks = [(i, 0.85) for i in range(1, 6)] + [(i, 0.80) for i in range(12, 17)]

        # Dynamic player tracks database populating
        for track_id, avg_conf in unique_tracks:
            if track_id == 99:
                team = "Referee"
            elif track_id % 2 == 0:
                team = "Team A"
            else:
                team = "Team B"
            db.add(PlayerTrack(job_id=job_id, track_id=track_id, team=team, confidence=float(avg_conf or 0.85)))
        db.commit()

    # Create one simple pass event dynamic to the actual tracks
    track_ids = [t[0] for t in unique_tracks if t[0] != 99]
    passer_id = track_ids[0] if len(track_ids) > 0 else 1
    receiver_id = track_ids[1] if len(track_ids) > 1 else 2

    p = PassEvent(
        job_id=job_id, passer_track_id=passer_id, receiver_track_id=receiver_id, 
        timestamp=2.0, outcome="completed", confidence=0.90
    )
    db.add(p)
    db.commit()

    db.add(PassingOption(
        pass_event_id=p.id, candidate_track_id=receiver_id, source="observed", 
        score=0.70, confidence=0.90, explanation="Standard square pass."
    ))
    db.commit()
