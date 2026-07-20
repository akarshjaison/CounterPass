import os
# pyrefly: ignore [missing-import]
import cv2
import math
import random
import numpy as np
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from app.models.models import PlayerDetection
from app.core.config import settings

def get_player_pos_at_frame(track_id: int, f: int, base_positions: dict) -> tuple:
    """
    Computes a player's simulated 2D position at a specific frame index.
    """
    bx, by = base_positions.get(track_id, (960, 540))
    t = f * 0.02
    dx = 150.0 * math.sin(t + track_id)
    dy = 80.0 * math.cos(t * 1.5 + track_id)
    return bx + dx, by + dy

def get_simulated_ball_position(f: int, fps: float, base_positions: dict) -> tuple:
    """
    Computes the ball's simulated position, interpolating between players to mirror mock pass events.
    """
    sec = f / fps
    
    # Ball passing sequence timeline:
    # 0s to 3s: Ball with Player 8 (Team A Midfielder)
    # 3s to 4.5s: Ball traveling from Player 8 to Player 10 (Pass 1)
    # 4.5s to 10s: Ball with Player 10
    # 10s to 12.2s: Ball traveling from Player 14 to Player 17 (Pass 2)
    # 12.2s to 18s: Ball with Player 17
    # 18s to 19.8s: Ball traveling from Player 10 to Player 7 (Pass 3)
    # 19.8s to 26s: Ball with Player 7
    # 26s to 28.5s: Ball traveling from Player 18 to Player 20 (Pass 4)
    # 28.5s+: Ball with Player 20
    segments = [
        (0.0, 3.0, 8, 8),
        (3.0, 4.5, 8, 10),
        (4.5, 10.0, 10, 10),
        (10.0, 12.2, 14, 17),
        (12.2, 18.0, 17, 17),
        (18.0, 19.8, 10, 7),
        (19.8, 26.0, 7, 7),
        (26.0, 28.5, 18, 20),
        (28.5, 999.0, 20, 20)
    ]
    
    for start, end, p1, p2 in segments:
        if start <= sec < end:
            x1, y1 = get_player_pos_at_frame(p1, f, base_positions)
            x2, y2 = get_player_pos_at_frame(p2, f, base_positions)
            
            duration = end - start
            if duration <= 0 or p1 == p2:
                return x1, y1
            
            t = (sec - start) / duration
            bx = x1 + (x2 - x1) * t
            by = y1 + (y2 - y1) * t
            return bx, by
            
    return 960, 540

def run_simulated_detection(db: Session, job_id: int, video_path: str, downsample_fps: float = 5.0):
    """
    Simulates a high-fidelity tactical broad view video stream, writing 
    frame-by-frame player and ball detections to the database.
    """
    cap = cv2.VideoCapture(video_path)
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    width_val = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_val = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count_val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    fps = float(fps_val) if fps_val and fps_val > 0 else 30.0
    width = int(width_val) if width_val and width_val > 0 else 1920
    height = int(height_val) if height_val and height_val > 0 else 1080
    frame_count = int(frame_count_val) if frame_count_val and frame_count_val > 0 else 300

    frame_interval = max(1, int(fps / downsample_fps))
    
    # Base layout coordinates matching teams and referee
    # Team A: 1-11, Team B: 12-22, Referee: 99
    base_positions = {
        1: (150, 540),
        2: (400, 200),
        3: (400, 400),
        4: (400, 680),
        5: (400, 880),
        6: (700, 300),
        7: (700, 540),
        8: (700, 780),
        9: (1000, 200),
        10: (1000, 540),
        11: (1000, 880),
        
        12: (1770, 540),
        13: (1520, 200),
        14: (1520, 400),
        15: (1520, 680),
        16: (1520, 880),
        17: (1220, 300),
        18: (1220, 540),
        19: (1220, 780),
        20: (920, 200),
        21: (920, 540),
        22: (920, 880),
        
        99: (960, 500),
    }

    detections = []
    
    # Consistent seeds for reproducible mocks
    random.seed(job_id)
    
    for f in range(0, frame_count, frame_interval):
        timestamp = f / fps
        
        # Save player & referee positions
        for track_id, (bx, by) in base_positions.items():
            x, y = get_player_pos_at_frame(track_id, f, base_positions)
            x = max(50, min(width - 50, x))
            y = max(50, min(height - 50, y))
            
            w = float(30 + random.randint(-2, 2))
            h = float(60 + random.randint(-4, 4))
            
            x_min = float(x - w / 2)
            y_min = float(y - h / 2)
            x_max = float(x + w / 2)
            y_max = float(y + h / 2)
            
            detections.append(PlayerDetection(
                job_id=job_id,
                frame_index=f,
                timestamp=timestamp,
                track_id=track_id,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                center_x=float(x),
                center_y=float(y),
                confidence=float(0.85 + 0.14 * random.random()),
                class_id=0 if track_id != 99 else 1
            ))
            
        # Save ball position
        bx, by = get_simulated_ball_position(f, fps, base_positions)
        bx = max(10, min(width - 10, bx))
        by = max(10, min(height - 10, by))
        
        detections.append(PlayerDetection(
            job_id=job_id,
            frame_index=f,
            timestamp=timestamp,
            track_id=None,
            x_min=float(bx - 10),
            y_min=float(by - 10),
            x_max=float(bx + 10),
            y_max=float(by + 10),
            center_x=float(bx),
            center_y=float(by),
            confidence=float(0.75 + 0.20 * random.random()),
            class_id=32
        ))
        
    db.add_all(detections)
    db.commit()

def run_yolo_detection(db: Session, job_id: int, video_path: str, weights_path: str, downsample_fps: float = 5.0):
    """
    Performs YOLOv8 frame inference using OpenCV's DNN module on a video file.
    Extracts person (class 0) and sports ball (class 32) detections.
    """
    from app.services.tracker import SimpleTracker, BallTracker
    tracker = SimpleTracker(max_lost_frames=settings.MAX_LOST_FRAMES)
    ball_tracker = BallTracker(max_lost_frames=10)
    net = cv2.dnn.readNetFromONNX(weights_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file at {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    
    frame_interval = max(1, int(fps / downsample_fps))
    
    detections = []
    track_colors = {}  # track_id -> list of (r, g, b) colors
    f = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if f % frame_interval == 0:
            timestamp = f / fps
            
            # Prepare standard 640x640 YOLO blob
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()
            
            # Parse YOLOv8 outputs: shape is (1, 84, 8400)
            if len(outputs.shape) == 3 and outputs.shape[1] == 84:
                output = outputs[0]
                cols = output.shape[1]
                
                boxes = []
                confidences = []
                class_ids = []
                
                for i in range(cols):
                    classes_scores = output[4:, i]
                    person_score = classes_scores[0]
                    ball_score = classes_scores[32] if len(classes_scores) > 32 else 0.0
                    
                    if person_score > 0.4:
                        cx, cy, w, h = output[0, i], output[1, i], output[2, i], output[3, i]
                        x = int((cx - w/2) * (width / 640.0))
                        y = int((cy - h/2) * (height / 640.0))
                        w = int(w * (width / 640.0))
                        h = int(h * (height / 640.0))
                        boxes.append([x, y, w, h])
                        confidences.append(float(person_score))
                        class_ids.append(0)
                    elif ball_score > 0.3:
                        cx, cy, w, h = output[0, i], output[1, i], output[2, i], output[3, i]
                        x = int((cx - w/2) * (width / 640.0))
                        y = int((cy - h/2) * (height / 640.0))
                        w = int(w * (width / 640.0))
                        h = int(h * (height / 640.0))
                        boxes.append([x, y, w, h])
                        confidences.append(float(ball_score))
                        class_ids.append(32)
                
                # Partition by class for independent NMS
                person_indices = [idx for idx, cid in enumerate(class_ids) if cid == 0]
                ball_indices = [idx for idx, cid in enumerate(class_ids) if cid == 32]
                
                p_boxes = [boxes[idx] for idx in person_indices]
                p_confs = [confidences[idx] for idx in person_indices]
                nms_p = cv2.dnn.NMSBoxes(p_boxes, p_confs, 0.4, 0.5)
                
                b_boxes = [boxes[idx] for idx in ball_indices]
                b_confs = [confidences[idx] for idx in ball_indices]
                nms_b = cv2.dnn.NMSBoxes(b_boxes, b_confs, 0.3, 0.5)
                
                tracker_inputs = []
                for idx in nms_p:
                    # nms_p may return array or list depending on opencv version
                    idx_val = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
                    actual_idx = person_indices[idx_val]
                    box = boxes[actual_idx]
                    conf = confidences[actual_idx]
                    
                    x_min = max(0.0, float(box[0]))
                    y_min = max(0.0, float(box[1]))
                    x_max = min(float(width), float(box[0] + box[2]))
                    y_max = min(float(height), float(box[1] + box[3]))
                    
                    tracker_inputs.append({
                        'box': [x_min, y_min, x_max, y_max],
                        'confidence': conf
                    })
                
                # Update tracker with detections
                tracked_players = tracker.update(tracker_inputs)
                
                # Save player/person detections
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
                    
                    # Extract jersey color for the chest crop
                    from app.services.classifier import extract_jersey_color
                    color = extract_jersey_color(frame, tp['box'])
                    tid = tp['id']
                    if tid not in track_colors:
                        track_colors[tid] = []
                    track_colors[tid].append(color)
                    
                # Process ball detections through BallTracker
                ball_inputs = []
                for idx in nms_b:
                    idx_val = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
                    actual_idx = ball_indices[idx_val]
                    box = boxes[actual_idx]
                    conf = confidences[actual_idx]
                    
                    x_min = max(0.0, float(box[0]))
                    y_min = max(0.0, float(box[1]))
                    x_max = min(float(width), float(box[0] + box[2]))
                    y_max = min(float(height), float(box[1] + box[3]))
                    
                    ball_inputs.append({
                        'box': [x_min, y_min, x_max, y_max],
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
        f += 1
        
    cap.release()
    
    db.add_all(detections)
    db.commit()
    
    # Calculate average color per track_id and run team clustering
    if track_colors:
        avg_track_colors = {}
        for tid, colors in track_colors.items():
            if colors:
                r_avg = sum(c[0] for c in colors) / len(colors)
                g_avg = sum(c[1] for c in colors) / len(colors)
                b_avg = sum(c[2] for c in colors) / len(colors)
                avg_track_colors[tid] = [r_avg, g_avg, b_avg]
                
        from app.services.classifier import kmeans_classify
        team_classifications = kmeans_classify(avg_track_colors)
        
        from app.models.models import PlayerTrack
        for tid, team in team_classifications.items():
            track_dets = [d for d in detections if d.track_id == tid]
            avg_conf = sum(d.confidence for d in track_dets) / len(track_dets) if track_dets else 0.85
            db.add(PlayerTrack(
                job_id=job_id,
                track_id=tid,
                team=team,
                confidence=float(avg_conf)
            ))
        db.commit()

def run_player_detection(db: Session, job_id: int, video_path: str, mode: str):
    """
    Main entry point for player and ball detection pipeline.
    Runs real CV detection if YOLOv8 weights are found, otherwise falls back gracefully to simulation.
    """
    weights_path = os.path.join(settings.BASE_DIR, "app", "models", "yolov8n.onnx")
    
    # Fallback to simulation if mode is demo or YOLO weights are missing
    if mode == "demo" or not os.path.exists(weights_path):
        run_simulated_detection(db, job_id, video_path)
    else:
        try:
            run_yolo_detection(db, job_id, video_path, weights_path)
        except Exception as e:
            # Degrade gracefully to simulated pipeline to keep backend functional
            print(f"YOLO detection error, falling back to simulation: {e}")
            run_simulated_detection(db, job_id, video_path)
