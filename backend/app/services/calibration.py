import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sqlalchemy.orm import Session
from app.models.models import PlayerDetection, PlayerTrack
from app.core.config import settings

# Attempt to import sports package for dynamic pitch calibration
try:
    from sports.configs.soccer import SoccerPitchConfiguration
    from sports.common.view import ViewTransformer
    SPORTS_AVAILABLE = True
except ImportError:
    SPORTS_AVAILABLE = False
    print("[Calibration] sports package not available. Dynamic homography disabled.")

class PitchCalibrator:
    def __init__(self, pitch_model_path: Optional[str] = None):
        self.transformer = None
        self.pitch_model = None
        
        if pitch_model_path and os.path.exists(pitch_model_path) and SPORTS_AVAILABLE:
            try:
                from ultralytics import YOLO
                self.pitch_model = YOLO(pitch_model_path)
            except ImportError:
                print("[Calibration] ultralytics not available.")
        
    def calibrate_from_frame(self, frame: np.ndarray, conf_threshold: float = 0.3) -> bool:
        """
        Detects pitch keypoints in the given frame and initializes the ViewTransformer.
        Returns True if successful, False otherwise.
        """
        if not self.pitch_model or not SPORTS_AVAILABLE:
            return False
            
        import supervision as sv
        results = self.pitch_model(frame, conf=conf_threshold, verbose=False)
        result = list(results)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)
        
        config = SoccerPitchConfiguration()
        
        if len(key_points.xy) > 0 and key_points.xy.shape[1] > 0:
            if key_points.confidence is not None and len(key_points.confidence) > 0:
                filter_mask = key_points.confidence[0] > 0.5
            else:
                # Fallback if model provides no confidence scores
                filter_mask = np.ones(key_points.xy.shape[1], dtype=bool)
                
            frame_reference_points = key_points.xy[0][filter_mask]
            pitch_reference_points = np.array(config.vertices)[filter_mask]
            
            if len(frame_reference_points) >= 4:
                self.transformer = ViewTransformer(
                    source=pitch_reference_points,
                    target=frame_reference_points
                )
                print("[Calibration] Pitch homography calculated successfully.")
                return True
        print("[Calibration] Not enough high-confidence pitch keypoints found.")
        return False
        
    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transforms a pixel point (x, y) into pitch coordinates (meters or cm, based on config).
        Returns (x, y) if calibration failed.
        """
        if self.transformer:
            # ViewTransformer.transform_points takes a numpy array of shape (N, 2)
            points = np.array([[x, y]], dtype=np.float32)
            transformed = self.transformer.transform_points(points=points)
            # transformed is typically (N, 2)
            return (float(transformed[0][0]), float(transformed[0][1]))
            
        # Fallback to simple mapping if uncalibrated
        return (x, y)

def infer_team_attack_directions(
    db: Session,
    job_id: int,
    calibrator: Optional[PitchCalibrator] = None,
    duration_secs: float = 30.0
) -> Dict[str, str]:
    """
    Determines team attack directions ("right" = attacking +x, "left" = attacking -x)
    by computing average x-positions over the first duration_secs of tracked data.
    Team positioned further left (smaller x) attacks to the right (+x).
    """
    tracks = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id).all()
    if not tracks:
        return {"Team A": "right", "Team B": "left"}

    team_x_coords: Dict[str, List[float]] = {}
    
    dets = db.query(PlayerDetection).filter(
        PlayerDetection.job_id == job_id,
        PlayerDetection.timestamp <= duration_secs,
        PlayerDetection.track_id.isnot(None),
        PlayerDetection.class_id == 0
    ).all()

    track_team_map = {t.track_id: t.team for t in tracks if t.team}

    for d in dets:
        team = track_team_map.get(d.track_id, "Unknown")  # type: ignore
        if team in ("Unknown", "Referee"):
            continue
        
        x_val = d.center_x
        if calibrator is not None and calibrator.transformer is not None:
            x_val, _ = calibrator.transform_point(d.center_x, d.center_y)  # type: ignore
            
        team_x_coords.setdefault(team, []).append(x_val)  # type: ignore

    teams = list(team_x_coords.keys())
    if len(teams) >= 2:
        avg_x_1 = sum(team_x_coords[teams[0]]) / len(team_x_coords[teams[0]]) if team_x_coords[teams[0]] else 50.0
        avg_x_2 = sum(team_x_coords[teams[1]]) / len(team_x_coords[teams[1]]) if team_x_coords[teams[1]] else 50.0

        if avg_x_1 < avg_x_2:
            return {teams[0]: "right", teams[1]: "left"}
        else:
            return {teams[0]: "left", teams[1]: "right"}

    return {"Team A": "right", "Team B": "left"}
