from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List, Optional

# Video Schemas
class VideoBase(BaseModel):
    filename: str
    path: str
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None

class VideoCreate(VideoBase):
    pass

class VideoResponse(VideoBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# AnalysisJob Schemas
class AnalysisJobBase(BaseModel):
    video_id: int
    status: str
    progress: float
    current_stage: Optional[str] = None
    error_message: Optional[str] = None

class AnalysisJobCreate(BaseModel):
    video_id: int

class AnalysisJobResponse(AnalysisJobBase):
    id: int
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

# PlayerTrack Schemas
class PlayerTrackBase(BaseModel):
    track_id: int
    team: str
    confidence: float

class PlayerTrackResponse(PlayerTrackBase):
    id: int
    job_id: int

    model_config = ConfigDict(from_attributes=True)

# PassingOption Schemas
class PassingOptionBase(BaseModel):
    candidate_track_id: int
    source: str  # observed, temporally_inferred
    score: float
    confidence: float
    explanation: Optional[str] = None

class PassingOptionResponse(PassingOptionBase):
    id: int
    pass_event_id: int
    x: Optional[float] = None
    y: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)

# Opponent coordinate schema
class OpponentPosition(BaseModel):
    id: int
    x: float
    y: float

# PassEvent Schemas
class PassEventBase(BaseModel):
    passer_track_id: int
    receiver_track_id: int
    timestamp: float
    outcome: str
    confidence: float

class PassEventResponse(PassEventBase):
    id: int
    job_id: int
    options: List[PassingOptionResponse] = []
    passer_x: Optional[float] = None
    passer_y: Optional[float] = None
    receiver_x: Optional[float] = None
    receiver_y: Optional[float] = None
    opponents: List[OpponentPosition] = []

    model_config = ConfigDict(from_attributes=True)

# MissedOpportunity Schemas
class MissedOpportunityBase(BaseModel):
    timestamp: float
    carrier_track_id: int
    recommended_track_id: int
    score: float
    confidence: float
    explanation: Optional[str] = None

class MissedOpportunityResponse(MissedOpportunityBase):
    id: int
    job_id: int

    model_config = ConfigDict(from_attributes=True)

# PlayerDetection Schemas
class PlayerDetectionBase(BaseModel):
    frame_index: int
    timestamp: float
    track_id: Optional[int] = None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    center_x: float
    center_y: float
    confidence: float
    class_id: int

class PlayerDetectionCreate(PlayerDetectionBase):
    pass

class PlayerDetectionResponse(PlayerDetectionBase):
    id: int
    job_id: int

    model_config = ConfigDict(from_attributes=True)


# Dashboard & Metrics Schemas
class GeneralMetrics(BaseModel):
    total_passes: int
    completed_passes: int
    completion_rate: float
    missed_opportunities_count: int
    forward_passes: int
    risky_passes: int
    avg_option_score: float
    counterpass_score: float  # Composite 0-100 score
    decision_making_rating: float
    awareness_rating: float
    positioning_rating: float
    movement_rating: float

class TimeSeriesInvolvement(BaseModel):
    timestamp: float
    team_a_possession: float
    team_b_possession: float
    active_players_count: int
    pressure_index: float

class JobDetailedResponse(BaseModel):
    job: AnalysisJobResponse
    video: VideoResponse
    metrics: Optional[GeneralMetrics] = None
    involvement_timeline: List[TimeSeriesInvolvement] = []

    model_config = ConfigDict(from_attributes=True)
