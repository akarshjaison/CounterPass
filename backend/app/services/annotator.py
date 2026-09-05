import os
import cv2
import math
import numpy as np
from typing import Dict, List, Tuple, Any
from sqlalchemy.orm import Session
from app.models.models import AnalysisJob, PlayerDetection, PassEvent, PlayerTrack
from app.core.config import settings
import supervision as sv

# Try importing sports for pitch lines
try:
    from sports.configs.soccer import SoccerPitchConfiguration
    SPORTS_AVAILABLE = True
except ImportError:
    SPORTS_AVAILABLE = False

def annotate_video(db: Session, job_id: int):
    """
    Renders an annotated tactical video using supervision for polished drawing.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job or not job.video:
        return
        
    video_path = job.video.path
    if not os.path.exists(video_path):
        return
        
    output_filename = f"job_{job_id}_annotated.webm"
    output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with open(output_path, "wb") as placeholder:
            placeholder.write(b"CV Annotator Graceful Placeholder Video Output")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
    
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        cap.release()
        return

    # Pitch Calibration for drawing lines
    frame_all_points = None
    if SPORTS_AVAILABLE:
        from app.services.calibration import PitchCalibrator
        pitch_model_path = os.path.join(settings.MODELS_DIR, "pose_field.pt")
        calibrator = PitchCalibrator(pitch_model_path)
        
        # Read first frame to get static homography
        ret, first_frame = cap.read()
        if ret and calibrator.calibrate_from_frame(first_frame):
            config = SoccerPitchConfiguration()
            pitch_all_points = np.array(config.vertices)
            
            # ViewTransformer goes Pitch -> Pixel if we reverse the source/target?
            # Actually transformer.transform_points() transforms source to target. 
            # We defined it as source=pitch_reference, target=frame_reference
            # So transforming pitch_all_points will give us frame pixels!
            frame_all_points = calibrator.transformer.transform_points(points=pitch_all_points)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset

    # Setup Annotators
    team_a_color = sv.Color.from_hex('#0DF27B')  # Neon green
    team_b_color = sv.Color.from_hex('#F8BD38')  # Cyan/Blue-ish Orange
    ref_color = sv.Color.from_hex('#00F2FF')     # Bright Cyan for ref
    ball_color = sv.Color.from_hex('#FFFFFF')    # White
    
    team_ellipse_annotator = sv.EllipseAnnotator(thickness=2)
    team_label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER)
    
    referee_ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette([ref_color]), thickness=2)
    referee_label_annotator = sv.LabelAnnotator(color=sv.ColorPalette([ref_color]), text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER)
    
    ball_triangle_annotator = sv.TriangleAnnotator(color=ball_color, base=25, height=21, outline_thickness=1)
    
    edge_annotator = None
    if SPORTS_AVAILABLE:
        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex('#FFFFFF'),
            thickness=2,
            edges=SoccerPitchConfiguration().edges
        )

    tracks = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id).all()
    team_map: Dict[int, str] = {int(t.track_id): str(t.team) for t in tracks}  # type: ignore
    passes = db.query(PassEvent).filter(PassEvent.job_id == job_id).all()
    
    detections = db.query(PlayerDetection).filter(PlayerDetection.job_id == job_id).all()
    detections_by_frame: Dict[int, List[PlayerDetection]] = {}
    for d in detections:
        f_idx = int(d.frame_index)  # type: ignore
        if f_idx not in detections_by_frame:
            detections_by_frame[f_idx] = []
        detections_by_frame[f_idx].append(d)

    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Draw Pitch Lines
        annotated_frame: Any = frame.copy()
        if SPORTS_AVAILABLE and edge_annotator is not None and frame_all_points is not None:
            # edge_annotator uses key_points
            kp = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
            annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=kp)

        frame_dets = detections_by_frame.get(f, [])
        ball = next((d for d in frame_dets if d.class_id == 32), None)
        players = [d for d in frame_dets if d.class_id == 0 and d.track_id is not None]
        
        # 2. Draw Players
        player_boxes = []
        player_tids = []
        player_class_ids = []
        
        ref_boxes = []
        ref_tids = []
        
        for p in players:
            team = team_map.get(p.track_id, "Team A")
            if team == "Referee":
                ref_boxes.append([p.x_min, p.y_min, p.x_max, p.y_max])
                ref_tids.append(p.track_id)
            else:
                player_boxes.append([p.x_min, p.y_min, p.x_max, p.y_max])
                player_tids.append(p.track_id)
                player_class_ids.append(0 if team == "Team A" else 1)
                
        # Annotate players
        if player_boxes:
            sv_dets = sv.Detections(
                xyxy=np.array(player_boxes, dtype=np.float32),
                tracker_id=np.array(player_tids, dtype=int),
                class_id=np.array(player_class_ids, dtype=int)
            )
            # Override colors to Team A/B
            team_ellipse_annotator.color = sv.ColorPalette([team_a_color, team_b_color])
            team_label_annotator.color = sv.ColorPalette([team_a_color, team_b_color])
            
            annotated_frame = team_ellipse_annotator.annotate(scene=annotated_frame, detections=sv_dets)
            labels = [f"P{tid}" for tid in player_tids]
            annotated_frame = team_label_annotator.annotate(scene=annotated_frame, detections=sv_dets, labels=labels)
            
        # Annotate Referees
        if ref_boxes:
            ref_dets = sv.Detections(
                xyxy=np.array(ref_boxes, dtype=np.float32),
                tracker_id=np.array(ref_tids, dtype=int),
                class_id=np.zeros(len(ref_boxes), dtype=int)
            )
            annotated_frame = referee_ellipse_annotator.annotate(scene=annotated_frame, detections=ref_dets)
            ref_labels = ["REF" for _ in ref_tids]
            annotated_frame = referee_label_annotator.annotate(scene=annotated_frame, detections=ref_dets, labels=ref_labels)

        # 3. Draw Ball
        if ball:
            ball_dets = sv.Detections(
                xyxy=np.array([[ball.x_min, ball.y_min, ball.x_max, ball.y_max]], dtype=np.float32),
                class_id=np.zeros(1, dtype=int)
            )
            annotated_frame = ball_triangle_annotator.annotate(scene=annotated_frame, detections=ball_dets)

        # 4. Draw passing lane overlays
        for p_ev in passes:
            start_frame = int(p_ev.timestamp * fps)  # type: ignore
            dur = 1.2 if p_ev.outcome == "completed" else 0.8
            end_frame = start_frame + int(dur * fps)
            
            if start_frame <= f <= end_frame:
                passer_det = next((p for p in players if p.track_id == p_ev.passer_track_id), None)
                receiver_det = next((p for p in players if p.track_id == p_ev.receiver_track_id), None)
                
                if passer_det:
                    px, py = int(passer_det.center_x), int(passer_det.center_y)  # type: ignore
                    cv2.circle(annotated_frame, (px, py), 20, (13, 242, 123), 2)
                    
                    if receiver_det:
                        rx, ry = int(receiver_det.center_x), int(receiver_det.center_y)  # type: ignore
                        lane_color = (13, 242, 123) if p_ev.outcome == "completed" else (0, 77, 255)
                        cv2.line(annotated_frame, (px, py), (rx, ry), lane_color, 3)
                        
                        angle = math.atan2(ry - py, rx - px)
                        arrow_size = 15
                        ax1 = int(rx - arrow_size * math.cos(angle - math.pi/6))
                        ay1 = int(ry - arrow_size * math.sin(angle - math.pi/6))
                        ax2 = int(rx - arrow_size * math.cos(angle + math.pi/6))
                        ay2 = int(ry - arrow_size * math.sin(angle + math.pi/6))
                        cv2.line(annotated_frame, (rx, ry), (ax1, ay1), lane_color, 3)
                        cv2.line(annotated_frame, (rx, ry), (ax2, ay2), lane_color, 3)

        writer.write(annotated_frame)
        f += 1
        
        if f % 30 == 0:
            job.progress = 90.0 + 9.0 * (f / max(1, frame_count))
            job.current_stage = f"Rendering Video ({f}/{frame_count})"
            db.commit()
        
    cap.release()
    writer.release()
