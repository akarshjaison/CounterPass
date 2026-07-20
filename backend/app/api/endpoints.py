import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import List

import math
from app.core.config import settings
from app.db.database import get_db
from app.models.models import Video, AnalysisJob, PlayerTrack, PassEvent, PassingOption, MissedOpportunity, PlayerDetection
from app.schemas import schemas
from app.utils.video import get_video_metadata
from app.services.analysis import start_analysis_job
from app.services.analytics_engine import compile_match_metrics

router = APIRouter()

@router.get("/health", response_model=dict)
def health_check():
    """
    Simple API health-check endpoint.
    """
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@router.post("/videos/upload", response_model=schemas.VideoResponse)
def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads a video, extracts metadata (FPS, dimensions, duration) using OpenCV,
    stores it in the uploads folder, and creates a database record.
    """
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid video format. Supported types: MP4, MOV, AVI, MKV"
        )
    
    # Save file
    filename = f"{int(datetime.now(timezone.utc).timestamp())}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")
        
    # Extract metadata
    try:
        meta = get_video_metadata(file_path)
    except Exception as e:
        # Cleanup file on metadata failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error reading video metadata: {str(e)}")
        
    # Create DB entry
    db_video = Video(
        filename=file.filename,
        path=file_path,
        fps=meta.get("fps"),
        width=meta.get("width"),
        height=meta.get("height"),
        duration=meta.get("duration")
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    
    return db_video

@router.post("/analysis/start/{video_id}", response_model=schemas.AnalysisJobResponse)
def start_analysis(video_id: int, config: schemas.AnalysisJobCreate, db: Session = Depends(get_db)):
    """
    Creates an analysis job (Real CV or Demo Mode) for the given video and triggers it in the background.
    """
    # Verify video exists
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    # Verify no active job for this video
    active_job = db.query(AnalysisJob).filter(
        AnalysisJob.video_id == video_id, 
        AnalysisJob.status.in_(["queued", "processing"])
    ).first()
    if active_job:
        return active_job
        
    # Create Job
    job = AnalysisJob(
        video_id=video_id,
        status="queued",
        progress=0.0,
        current_stage="Queued",
        mode=config.mode
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Trigger background worker
    start_analysis_job(job.id, mode=config.mode)
    
    return job

@router.get("/analysis/{job_id}/status", response_model=schemas.AnalysisJobResponse)
def get_job_status(job_id: int, db: Session = Depends(get_db)):
    """
    Queries current job progress and status.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return job

@router.get("/analysis/{job_id}/results", response_model=schemas.GeneralMetrics)
def get_job_results(job_id: int, db: Session = Depends(get_db)):
    """
    Generates summary performance and decision metrics for the completed analysis.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed yet. Current status: {job.status}")
        
    # Compile performance metrics based on the mode
    if job.mode == "demo":
        return schemas.GeneralMetrics(
            total_passes=5,
            completed_passes=3,
            completion_rate=60.0,
            missed_opportunities_count=2,
            forward_passes=3,
            risky_passes=2,
            avg_option_score=0.69,
            counterpass_score=78.0,
            decision_making_rating=71.0,
            awareness_rating=78.0,
            positioning_rating=80.0,
            movement_rating=75.0
        )
    else:
        metrics_dict = compile_match_metrics(db, job_id)
        return schemas.GeneralMetrics(**metrics_dict)

@router.get("/analysis/{job_id}/events", response_model=List[schemas.PassEventResponse])
def get_pass_events(job_id: int, db: Session = Depends(get_db)):
    """
    Returns list of passing events alongside player options for each event.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    events = db.query(PassEvent).filter(PassEvent.job_id == job_id).all()
    
    if job.mode == "demo":
        return events

    width = job.video.width or 1920
    height = job.video.height or 1080
    fps = job.video.fps or 30.0

    response_events = []
    for e in events:
        frame_idx = int(e.timestamp * fps + 0.1)
        
        # Query player detections at this frame
        dets = db.query(PlayerDetection).filter(
            PlayerDetection.job_id == job_id,
            PlayerDetection.frame_index == frame_idx
        ).all()
        
        # Track ID map to detection
        dets_map = {d.track_id: d for d in dets if d.track_id is not None}
        
        passer_det = dets_map.get(e.passer_track_id)
        receiver_det = dets_map.get(e.receiver_track_id)
        
        passer_x = (passer_det.center_x / width * 100.0) if passer_det else None
        passer_y = (passer_det.center_y / height * 100.0) if passer_det else None
        receiver_x = (receiver_det.center_x / width * 100.0) if receiver_det else None
        receiver_y = (receiver_det.center_y / height * 100.0) if receiver_det else None
        
        # Map options
        response_options = []
        for opt in e.options:
            cand_det = dets_map.get(opt.candidate_track_id)
            opt_x = (cand_det.center_x / width * 100.0) if cand_det else None
            opt_y = (cand_det.center_y / height * 100.0) if cand_det else None
            
            response_options.append(schemas.PassingOptionResponse(
                id=opt.id,
                pass_event_id=opt.pass_event_id,
                candidate_track_id=opt.candidate_track_id,
                source=opt.source,
                score=opt.score,
                confidence=opt.confidence,
                explanation=opt.explanation,
                x=opt_x,
                y=opt_y
            ))
            
        # Identify passer team
        passer_track = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id, PlayerTrack.track_id == e.passer_track_id).first()
        passer_team = passer_track.team if passer_track else ("Team A" if e.passer_track_id <= 11 else "Team B")
        
        # Map opponent positions
        opponents = []
        for d in dets:
            if d.class_id == 0 and d.track_id is not None:
                opp_track = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id, PlayerTrack.track_id == d.track_id).first()
                opp_team = opp_track.team if opp_track else ("Team A" if d.track_id <= 11 else "Team B")
                if opp_team != passer_team and opp_team != "Referee":
                    opponents.append(schemas.OpponentPosition(
                        id=d.track_id,
                        x=d.center_x / width * 100.0,
                        y=d.center_y / height * 100.0
                    ))
                    
        response_events.append(schemas.PassEventResponse(
            id=e.id,
            job_id=e.job_id,
            passer_track_id=e.passer_track_id,
            receiver_track_id=e.receiver_track_id,
            timestamp=e.timestamp,
            outcome=e.outcome,
            confidence=e.confidence,
            options=response_options,
            passer_x=passer_x,
            passer_y=passer_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            opponents=opponents
        ))
        
    return response_events

@router.get("/analysis/{job_id}/players", response_model=List[schemas.PlayerTrackResponse])
def get_players_list(job_id: int, db: Session = Depends(get_db)):
    """
    Returns tracked players classified by teams.
    """
    players = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id).all()
    return players

@router.get("/analysis/{job_id}/metrics", response_model=schemas.JobDetailedResponse)
def get_full_metrics_details(job_id: int, db: Session = Depends(get_db)):
    """
    Gathers detailed statistics and time-series data for dashboard visualization.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    video = db.query(Video).filter(Video.id == job.video_id).first()
    
    # Generate timeline
    timeline = []
    if job.status == "completed":
        if job.mode == "demo":
            timestamps = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
            team_a = [50.0, 80.0, 75.0, 30.0, 40.0, 20.0, 65.0, 70.0, 85.0, 50.0]
            team_b = [50.0, 20.0, 25.0, 70.0, 60.0, 80.0, 35.0, 30.0, 15.0, 50.0]
            players = [22, 22, 21, 22, 22, 20, 22, 22, 22, 22]
            pressure = [10.0, 15.0, 20.0, 40.0, 45.0, 35.0, 25.0, 15.0, 10.0, 20.0]
            
            for i in range(len(timestamps)):
                timeline.append(schemas.TimeSeriesInvolvement(
                    timestamp=timestamps[i],
                    team_a_possession=team_a[i],
                    team_b_possession=team_b[i],
                    active_players_count=players[i],
                    pressure_index=pressure[i]
                ))
        else:
            fps = video.fps or 30.0
            duration = video.duration or 30.0
            if duration <= 0 and video.fps and video.frame_count:
                duration = video.frame_count / video.fps
            if duration <= 0:
                duration = 45.0
                
            num_intervals = 10
            interval_len = duration / num_intervals
            
            all_dets = db.query(PlayerDetection).filter(PlayerDetection.job_id == job_id).all()
            tracks = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id).all()
            team_map = {t.track_id: t.team for t in tracks}
            
            for idx in range(num_intervals):
                t_start = idx * interval_len
                t_end = (idx + 1) * interval_len
                t_mid = (t_start + t_end) / 2.0
                
                window_dets = [d for d in all_dets if t_start <= d.timestamp < t_end]
                frames_in_win = set(d.frame_index for d in window_dets)
                
                avg_players = 22
                if frames_in_win:
                    player_dets = [d for d in window_dets if d.class_id == 0]
                    avg_players = int(len(player_dets) / len(frames_in_win) + 0.5)
                    
                team_a_frames = 0
                team_b_frames = 0
                
                dets_by_f = {}
                for d in window_dets:
                    if d.frame_index not in dets_by_f:
                        dets_by_f[d.frame_index] = []
                    dets_by_f[d.frame_index].append(d)
                    
                for f_idx, f_dets in dets_by_f.items():
                    ball = next((d for d in f_dets if d.class_id == 32), None)
                    players_f = [d for d in f_dets if d.class_id == 0 and d.track_id is not None]
                    if ball and players_f:
                        closest_p = min(players_f, key=lambda p: (p.center_x - ball.center_x)**2 + (p.center_y - ball.center_y)**2)
                        p_team = team_map.get(closest_p.track_id, "Team A" if closest_p.track_id <= 11 else "Team B")
                        if p_team == "Team A":
                            team_a_frames += 1
                        elif p_team == "Team B":
                            team_b_frames += 1
                            
                tot_pos = team_a_frames + team_b_frames
                if tot_pos > 0:
                    pos_a = (team_a_frames / tot_pos) * 100.0
                    pos_b = (team_b_frames / tot_pos) * 100.0
                else:
                    pos_a = 50.0 + 10.0 * math.sin(idx)
                    pos_b = 100.0 - pos_a
                    
                opp_dists = []
                for f_idx, f_dets in dets_by_f.items():
                    ball = next((d for d in f_dets if d.class_id == 32), None)
                    players_f = [d for d in f_dets if d.class_id == 0 and d.track_id is not None]
                    if ball and players_f:
                        closest_p = min(players_f, key=lambda p: (p.center_x - ball.center_x)**2 + (p.center_y - ball.center_y)**2)
                        p_team = team_map.get(closest_p.track_id, "Team A" if closest_p.track_id <= 11 else "Team B")
                        opps = [p for p in players_f if team_map.get(p.track_id, "Team A" if p.track_id <= 11 else "Team B") != p_team]
                        if opps:
                            min_opp_dist = min(math.sqrt((p.center_x - ball.center_x)**2 + (p.center_y - ball.center_y)**2) for p in opps)
                            opp_dists.append(min_opp_dist)
                            
                avg_opp_dist = (sum(opp_dists) / len(opp_dists)) if opp_dists else 120.0
                pressure_idx = max(5.0, min(95.0, 100.0 - (avg_opp_dist / 300.0 * 100.0)))
                
                timeline.append(schemas.TimeSeriesInvolvement(
                    timestamp=round(t_mid, 1),
                    team_a_possession=round(pos_a, 1),
                    team_b_possession=round(pos_b, 1),
                    active_players_count=min(22, max(2, avg_players)),
                    pressure_index=round(pressure_idx, 1)
                ))

    # Retrieve general metrics
    metrics_response = None
    if job.status == "completed":
        metrics_response = get_job_results(job_id, db)

    return schemas.JobDetailedResponse(
        job=schemas.AnalysisJobResponse.model_validate(job),
        video=schemas.VideoResponse.model_validate(video),
        metrics=metrics_response,
        involvement_timeline=timeline
    )

@router.get("/analysis/{job_id}/video")
def get_analyzed_video(job_id: int, db: Session = Depends(get_db)):
    """
    Returns the video file for playback. Serves the annotated CV video if it exists, otherwise falls back to source video.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    video = db.query(Video).filter(Video.id == job.video_id).first()
    if not video or not os.path.exists(video.path):
        raise HTTPException(status_code=404, detail="Video file does not exist on disk")
        
    # Check if annotated video exists in outputs
    annotated_filename = f"job_{job_id}_annotated.mp4"
    annotated_path = os.path.join(settings.OUTPUT_DIR, annotated_filename)
    if os.path.exists(annotated_path):
        return FileResponse(
            annotated_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"}
        )
        
    return FileResponse(
        video.path, 
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )

@router.get("/analysis/{job_id}/detections", response_model=List[schemas.PlayerDetectionResponse])
def get_player_detections(job_id: int, db: Session = Depends(get_db)):
    """
    Returns list of frame-by-frame player and ball detections for the job, sorted by frame_index.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    detections = db.query(PlayerDetection).filter(PlayerDetection.job_id == job_id).order_by(PlayerDetection.frame_index.asc()).all()
    return detections
