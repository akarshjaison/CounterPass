import os
import math
import cv2
import numpy as np
from typing import Dict, List
from sqlalchemy.orm import Session
from app.models.models import PlayerDetection, PlayerTrack
from app.core.config import settings
import gdown


def _ensure_model_weights(url: str, output_path: str) -> None:
    """
    Ensures the model weights file exists by downloading it via gdown if necessary.
    """
    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[Detection] Downloading model from {url} to {output_path} ...")
    try:
        gdown.download(url, output_path, quiet=False)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Failed to download to {output_path}")
        print(f"[Detection] Download complete: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download model weights: {e}")


def _track_avg_speed(track_dets: List[PlayerDetection]) -> float:
    ordered = sorted(track_dets, key=lambda d: d.timestamp)
    speeds = []
    for i in range(1, len(ordered)):
        dt = ordered[i].timestamp - ordered[i - 1].timestamp
        if dt <= 0:
            continue
        dx = ordered[i].center_x - ordered[i - 1].center_x
        dy = ordered[i].center_y - ordered[i - 1].center_y
        speeds.append(math.sqrt(dx * dx + dy * dy) / dt)
    return sum(speeds) / len(speeds) if speeds else 0.0


def _identify_static_tracks(dets_by_track: Dict[int, List[PlayerDetection]]) -> set:
    MIN_SAMPLES = 20
    speeds = {tid: _track_avg_speed(dets) for tid, dets in dets_by_track.items()}
    eligible_speeds = [s for tid, s in speeds.items() if len(dets_by_track[tid]) >= MIN_SAMPLES]
    if not eligible_speeds:
        return set()

    sorted_speeds = sorted(eligible_speeds)
    median_speed = sorted_speeds[len(sorted_speeds) // 2]
    if median_speed <= 0:
        return set()

    static_ids = set()
    for tid, dets in dets_by_track.items():
        if len(dets) < MIN_SAMPLES:
            continue
        if speeds[tid] < 0.15 * median_speed:
            static_ids.add(tid)
    return static_ids


def run_specialized_detection(db: Session, job_id: int, video_path: str, player_model_path: str, ball_model_path: str, downsample_fps: float = 5.0, start_pct: float = 15.0, end_pct: float = 35.0):
    from ultralytics import YOLO
    from app.services.tracker import SimpleTracker, BallTracker
    from app.services.classifier import extract_jersey_color, kmeans_classify

    tracker = SimpleTracker(max_lost_frames=settings.MAX_LOST_FRAMES)
    ball_tracker = BallTracker(max_lost_frames=10)
    
    print("[Detection] Loading specialized YOLO models...")
    player_model = YOLO(player_model_path)
    ball_model = YOLO(ball_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    
    frame_interval = max(1, int(fps / downsample_fps))
    
    detections = []
    track_colors = {}
    track_classes = {}
    f = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    
    while True:
        if f > 0 and f % max(30, int(fps)) == 0:
            from app.models.models import AnalysisJob
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                # Interpolate progress based on frames processed
                progress = start_pct + ((f / total_frames) * (end_pct - start_pct))
                job.progress = min(progress, end_pct)
                db.commit()
                
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = f / fps
        if f % frame_interval == 0:
            # 1. Player Detection
            player_results = player_model(frame, imgsz=480, verbose=False, conf=0.35)[0]
            
            tracker_inputs = []
            for box in player_results.boxes:
                cls_id = int(box.cls[0])
                cls_name = player_model.names[cls_id]
                conf = float(box.conf[0])
                
                if cls_name not in ['player', 'goalkeeper', 'referee']:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                color = extract_jersey_color(frame, [x1, y1, x2, y2])
                tracker_inputs.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'color': color,
                    'cls_name': cls_name
                })
            
            tracked_players = tracker.update(tracker_inputs)
            
            for tp in tracked_players:
                detections.append(PlayerDetection(
                    job_id=job_id,
                    frame_index=f,
                    timestamp=timestamp,
                    track_id=tp['id'],
                    x_min=tp['box'][0],
                    y_min=tp['box'][1],
                    x_max=tp['box'][2],
                    y_max=tp['box'][3],
                    center_x=tp['center'][0],
                    center_y=tp['center'][1],
                    confidence=tp['confidence'],
                    class_id=0
                ))
                tid = tp['id']
                color = tp.get('avg_color')
                cls_name = tp.get('cls_name', 'player')
                
                if color is not None:
                    track_colors.setdefault(tid, []).append(color)
                track_classes[tid] = cls_name
                    
            # 2. Ball Detection
            ball_results = ball_model(frame, imgsz=640, verbose=False, conf=settings.BALL_CONFIDENCE_THRESHOLD)[0]
            
            ball_inputs = []
            for box in ball_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                ball_inputs.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': conf
                })
            
            tracked_ball = ball_tracker.update(ball_inputs)
            if tracked_ball:
                detections.append(PlayerDetection(
                    job_id=job_id,
                    frame_index=f,
                    timestamp=timestamp,
                    track_id=None,
                    x_min=tracked_ball['box'][0],
                    y_min=tracked_ball['box'][1],
                    x_max=tracked_ball['box'][2],
                    y_max=tracked_ball['box'][3],
                    center_x=tracked_ball['center'][0],
                    center_y=tracked_ball['center'][1],
                    confidence=tracked_ball['confidence'],
                    class_id=32
                ))
        else:
            # Intermediate frames
            tracked_players = tracker.update([])
            for tp in tracked_players:
                detections.append(PlayerDetection(
                    job_id=job_id,
                    frame_index=f,
                    timestamp=timestamp,
                    track_id=tp['id'],
                    x_min=tp['box'][0],
                    y_min=tp['box'][1],
                    x_max=tp['box'][2],
                    y_max=tp['box'][3],
                    center_x=tp['center'][0],
                    center_y=tp['center'][1],
                    confidence=max(0.1, tp['confidence'] * 0.96),
                    class_id=0
                ))
            tracked_ball = ball_tracker.update([])
            if tracked_ball:
                detections.append(PlayerDetection(
                    job_id=job_id,
                    frame_index=f,
                    timestamp=timestamp,
                    track_id=None,
                    x_min=tracked_ball['box'][0],
                    y_min=tracked_ball['box'][1],
                    x_max=tracked_ball['box'][2],
                    y_max=tracked_ball['box'][3],
                    center_x=tracked_ball['center'][0],
                    center_y=tracked_ball['center'][1],
                    confidence=max(0.1, tracked_ball['confidence'] * 0.96),
                    class_id=32
                ))
        f += 1
        
    cap.release()
    db.add_all(detections)
    db.commit()
    
    # Team clustering logic
    if track_colors:
        dets_by_track: Dict[int, List[PlayerDetection]] = {}
        for d in detections:
            if d.track_id is not None:
                dets_by_track.setdefault(d.track_id, []).append(d)

        static_tids = _identify_static_tracks(dets_by_track)

        avg_track_colors = {}
        pre_assigned_teams = {}
        
        for tid, colors in track_colors.items():
            if tid in static_tids:
                continue
                
            cls_name = track_classes.get(tid, 'player')
            if cls_name == 'referee':
                pre_assigned_teams[tid] = 'Referee'
                continue
            elif cls_name == 'goalkeeper':
                pre_assigned_teams[tid] = 'Goalkeeper'
                # Alternatively we could cluster GK, but usually they wear different colors.
                # Safe to skip from main player clustering.
                continue
                
            if colors:
                r_avg = sum(c[0] for c in colors) / len(colors)
                g_avg = sum(c[1] for c in colors) / len(colors)
                b_avg = sum(c[2] for c in colors) / len(colors)
                avg_track_colors[tid] = [r_avg, g_avg, b_avg]

        from app.services.classifier import classify_teams_siglip
        # Classify remaining players (Team A / Team B)
        # classify_teams_siglip will fall back to KMeans on avg_colors if SigLIP is unavailable.
        team_classifications = classify_teams_siglip(avg_track_colors)
        
        # Merge pre-assigned Referees and Goalkeepers
        for tid, team in pre_assigned_teams.items():
            team_classifications[tid] = team
        
        team_tracks = {}
        for tid, team in team_classifications.items():
            track_dets = dets_by_track.get(tid, [])
            team_tracks.setdefault(team, []).append((tid, len(track_dets)))
            
        canonical_tids = set()
        for team, tracks_list in team_tracks.items():
            tracks_list.sort(key=lambda x: x[1], reverse=True)
            # Allow Referees and GKs, but limit Team A / Team B to 11
            limit = 11 if team in ['Team A', 'Team B'] else 5
            for tid, _ in tracks_list[:limit]:
                canonical_tids.add(tid)

        for tid in canonical_tids:
            team = team_classifications[tid]
            track_dets = dets_by_track.get(tid, [])
            avg_conf = sum(d.confidence for d in track_dets) / len(track_dets) if track_dets else 0.85
            db.add(PlayerTrack(
                job_id=job_id,
                track_id=tid,
                team=team,
                confidence=float(avg_conf)
            ))
        db.commit()


def run_player_detection(db: Session, job_id: int, video_path: str, start_pct: float = 15.0, end_pct: float = 35.0):
    """
    Main entry point for player and ball detection pipeline.
    """
    player_model_path = os.path.join(settings.MODELS_DIR, "detect_players.pt")
    ball_model_path = os.path.join(settings.MODELS_DIR, "detect_ball.pt")
    
    try:
        _ensure_model_weights(settings.PLAYER_MODEL_URL, player_model_path)
        _ensure_model_weights(settings.BALL_MODEL_URL, ball_model_path)
        run_specialized_detection(db, job_id, video_path, player_model_path, ball_model_path, start_pct=start_pct, end_pct=end_pct)
    except Exception as e:
        print(f"[Detection] Specialized detection failed: {e}.")
        raise
