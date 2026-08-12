import os
import math
import cv2
import numpy as np
from typing import Dict, List
from sqlalchemy.orm import Session
from app.models.models import PlayerDetection
from app.core.config import settings


def _track_avg_speed(track_dets: List[PlayerDetection]) -> float:
    """
    Average frame-to-frame center displacement speed (pixels/sec) for a track.
    Used to distinguish players (who move continuously) from largely-stationary
    sideline staff/cameramen picked up by the generic person detector.
    """
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
    """
    Flags tracks that are very unlikely to be players: cameramen, sideline staff,
    ball boys, or a stray non-pitch person picked up by the generic COCO 'person'
    class. Real players move continuously (jogging/sprinting) over the course of
    a match; someone standing on the touchline mostly doesn't.

    This uses a threshold *relative to the video's own median player speed*
    (not a fixed pixel value) so it adapts to camera zoom/resolution, and only
    acts on tracks with enough samples to make the call reliably.
    """
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

PT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt")

def _ensure_yolo_weights(weights_path: str) -> None:
    """
    Ensures the YOLOv8n ONNX weights file exists.
    Exports it from the local yolov8n.pt using the ultralytics library.
    """
    if os.path.exists(weights_path):
        return

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    pt_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt"))

    if not os.path.exists(pt_path):
        raise RuntimeError(
            f"YOLOv8n weights not found at {pt_path}. "
            "Please place yolov8n.pt in backend/app/models/."
        )

    print(f"[Detection] Exporting {pt_path} -> {weights_path} (ONNX) ...")
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO(pt_path)
        model.export(format="onnx", imgsz=640, opset=12)
        # ultralytics exports next to the .pt file; move it to weights_path
        exported = pt_path.replace(".pt", ".onnx")
        if os.path.exists(exported) and exported != weights_path:
            import shutil
            shutil.move(exported, weights_path)
        print(f"[Detection] Export complete: {weights_path}")
    except ImportError:
        raise RuntimeError(
            "ultralytics package is required to export ONNX weights. "
            "Run: pip install ultralytics"
        )


def run_yolo_detection(db: Session, job_id: int, video_path: str, weights_path: str, downsample_fps: float = 15.0):
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
            
        timestamp = f / fps
        if f % frame_interval == 0:
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
                    
                    if person_score > 0.35:
                        cx, cy, w, h = output[0, i], output[1, i], output[2, i], output[3, i]
                        x = int((cx - w/2) * (width / 640.0))
                        y = int((cy - h/2) * (height / 640.0))
                        w = int(w * (width / 640.0))
                        h = int(h * (height / 640.0))
                        boxes.append([x, y, w, h])
                        confidences.append(float(person_score))
                        class_ids.append(0)
                    elif ball_score > settings.BALL_CONFIDENCE_THRESHOLD:
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
                nms_p = cv2.dnn.NMSBoxes(p_boxes, p_confs, 0.35, 0.45)
                
                b_boxes = [boxes[idx] for idx in ball_indices]
                b_confs = [confidences[idx] for idx in ball_indices]
                nms_b = cv2.dnn.NMSBoxes(b_boxes, b_confs, settings.BALL_CONFIDENCE_THRESHOLD, 0.5)
                
                from app.services.classifier import extract_jersey_color

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
                    
                    # Compute jersey color up front so the tracker can use it as an
                    # appearance cue during identity matching (not just after the fact).
                    color = extract_jersey_color(frame, [x_min, y_min, x_max, y_max])
                    
                    tracker_inputs.append({
                        'box': [x_min, y_min, x_max, y_max],
                        'confidence': conf,
                        'color': color
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
                    
                    tid = tp['id']
                    color = tp.get('avg_color')
                    if color is not None:
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
        else:
            # Intermediate frame: interpolate/predict trajectories using trackers
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
    
    # Calculate average color per track_id and run team clustering
    if track_colors:
        from app.models.models import PlayerTrack

        # Group detections by track_id in a single pass O(N)
        dets_by_track: Dict[int, List[PlayerDetection]] = {}
        for d in detections:
            if d.track_id is not None:
                dets_by_track.setdefault(d.track_id, []).append(d)

        # Exclude tracks that barely move over a long span — most likely
        # cameramen/sideline staff, not players — *before* they can influence
        # team-color clustering or occupy a "top 11" canonical squad slot.
        static_tids = _identify_static_tracks(dets_by_track)

        avg_track_colors = {}
        for tid, colors in track_colors.items():
            if tid in static_tids:
                continue
            if colors:
                r_avg = sum(c[0] for c in colors) / len(colors)
                g_avg = sum(c[1] for c in colors) / len(colors)
                b_avg = sum(c[2] for c in colors) / len(colors)
                avg_track_colors[tid] = [r_avg, g_avg, b_avg]

        from app.services.classifier import kmeans_classify
        team_classifications = kmeans_classify(avg_track_colors)

        # Group track IDs by team and count detections
        team_tracks = {}
        for tid, team in team_classifications.items():
            track_dets = dets_by_track.get(tid, [])
            team_tracks.setdefault(team, []).append((tid, len(track_dets)))
            
        # Retain at most top 11 players per team (canonical squad of 22)
        canonical_tids = set()
        for team, tracks_list in team_tracks.items():
            tracks_list.sort(key=lambda x: x[1], reverse=True)
            for tid, _ in tracks_list[:11]:
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

def run_player_detection(db: Session, job_id: int, video_path: str):
    """
    Main entry point for player and ball detection pipeline.
    Always runs real YOLOv8 ONNX detection if possible.
    """
    weights_path = os.path.join(settings.BASE_DIR, "app", "models", "yolov8n.onnx")
    try:
        _ensure_yolo_weights(weights_path)
        run_yolo_detection(db, job_id, video_path, weights_path)
    except Exception as e:
        print(f"[Detection] YOLOv8 detection failed: {e}. Gracefully skipping CV detection.")
        pass
