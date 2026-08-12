import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "CounterPass"
    API_V1_STR: str = "/api"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    
    # Database
    DATABASE_URL: str = "sqlite:///" + os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "counterpass.db")
    
    # Directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
    
    # Temporal Analytics Thresholds
    DECAY_LAMBDA: float = 0.5  # Confidence decay parameter for temporally inferred options (confidence = orig * exp(-lambda * time))
    MAX_LOST_FRAMES: int = 30  # Number of frames to retain lost tracks
    TEMPORAL_BUFFER_SIZE: int = 60  # Number of frames to store in the sliding window buffer
    POSSESSION_DISTANCE_THRESHOLD: float = 100.0  # Pixels in frame coordinates or cm in pitch coordinates
    LANE_SAFETY_RADIUS: float = 25.0  # Danger radius for lane interception
    BALL_CONFIDENCE_THRESHOLD: float = 0.25  # Confidence threshold for ball detections to cut false positives
    
    # Option Scoring Weights
    WEIGHT_LANE_CLEARANCE: float = 0.25
    WEIGHT_SPACE_SCORE: float = 0.20
    WEIGHT_PROGRESSION_VALUE: float = 0.25
    WEIGHT_MOVEMENT_SCORE: float = 0.15
    WEIGHT_DISTANCE_SCORE: float = 0.15

    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
# Also place .gitkeep files to satisfy git commits
with open(os.path.join(settings.UPLOAD_DIR, ".gitkeep"), "a") as f:
    pass
with open(os.path.join(settings.OUTPUT_DIR, ".gitkeep"), "a") as f:
    pass
