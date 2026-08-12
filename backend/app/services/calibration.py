import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any
from sqlalchemy.orm import Session
from app.models.models import PlayerDetection, PlayerTrack

# Standard pitch dimensions in meters
PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0

DEFAULT_DST_POINTS = np.float32([
    [0.0, 0.0],
    [PITCH_LENGTH_METERS, 0.0],
    [PITCH_LENGTH_METERS, PITCH_WIDTH_METERS],
    [0.0, PITCH_WIDTH_METERS]
])

def compute_homography_matrix(
    src_points: Optional[List[Tuple[float, float]]] = None,
    img_width: int = 1920,
    img_height: int = 1080
) -> np.ndarray:
    """
    Computes a 3x3 homography matrix mapping pixel coordinates to pitch meters (105m x 68m).
    If src_points is not provided, defaults to full-frame corner correspondences.
    """
    if not src_points or len(src_points) != 4:
        # Default perspective approximation mapping 1920x1080 frame bounds
        src = np.float32([
            [0, 0],
            [img_width, 0],
            [img_width, img_height],
            [0, img_height]
        ])
    else:
        src = np.float32(src_points)

    H, _ = cv2.findHomography(src, DEFAULT_DST_POINTS)
    if H is None:
        H = np.eye(3, dtype=np.float32)
    return H

def transform_point(x: float, y: float, H: np.ndarray) -> Tuple[float, float]:
    """
    Transforms a (x, y) point from pixel space to pitch meter space using homography matrix H.
    """
    point = np.array([x, y, 1.0], dtype=np.float32)
    transformed = H @ point
    if transformed[2] != 0:
        tx = float(transformed[0] / transformed[2])
        ty = float(transformed[1] / transformed[2])
        return (tx, ty)
    return (x, y)

def infer_team_attack_directions(
    db: Session,
    job_id: int,
    H: Optional[np.ndarray] = None,
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
    
    # Query detections in the first 30 seconds
    dets = db.query(PlayerDetection).filter(
        PlayerDetection.job_id == job_id,
        PlayerDetection.timestamp <= duration_secs,
        PlayerDetection.track_id.isnot(None),
        PlayerDetection.class_id == 0
    ).all()

    track_team_map = {t.track_id: t.team for t in tracks if t.team}

    for d in dets:
        team = track_team_map.get(d.track_id, "Unknown")
        if team in ("Unknown", "Referee"):
            continue
        
        x_val = d.center_x
        if H is not None:
            x_val, _ = transform_point(d.center_x, d.center_y, H)
            
        team_x_coords.setdefault(team, []).append(x_val)

    teams = list(team_x_coords.keys())
    if len(teams) >= 2:
        avg_x_1 = sum(team_x_coords[teams[0]]) / len(team_x_coords[teams[0]]) if team_x_coords[teams[0]] else 50.0
        avg_x_2 = sum(team_x_coords[teams[1]]) / len(team_x_coords[teams[1]]) if team_x_coords[teams[1]] else 50.0

        if avg_x_1 < avg_x_2:
            return {teams[0]: "right", teams[1]: "left"}
        else:
            return {teams[0]: "left", teams[1]: "right"}

    return {"Team A": "right", "Team B": "left"}
