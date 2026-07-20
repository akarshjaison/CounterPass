from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.db.database import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    jobs = relationship("AnalysisJob", back_populates="video", cascade="all, delete-orphan")

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    status = Column(String, default="uploaded")  # uploaded, queued, processing, completed, failed
    progress = Column(Float, default=0.0)         # 0.0 to 100.0
    current_stage = Column(String, nullable=True) # e.g., "Extracting Frames", "Detecting Players"
    error_message = Column(String, nullable=True)
    mode = Column(String, default="real")         # real, demo
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    video = relationship("Video", back_populates="jobs")
    tracks = relationship("PlayerTrack", back_populates="job", cascade="all, delete-orphan")
    passes = relationship("PassEvent", back_populates="job", cascade="all, delete-orphan")
    missed_opportunities = relationship("MissedOpportunity", back_populates="job", cascade="all, delete-orphan")
    detections = relationship("PlayerDetection", back_populates="job", cascade="all, delete-orphan")

class PlayerTrack(Base):
    __tablename__ = "player_tracks"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    track_id = Column(Integer, nullable=False)
    team = Column(String, default="Unknown")      # Team A, Team B, Unknown, Referee
    confidence = Column(Float, default=1.0)

    job = relationship("AnalysisJob", back_populates="tracks")

class PassEvent(Base):
    __tablename__ = "pass_events"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    passer_track_id = Column(Integer, nullable=False)
    receiver_track_id = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)     # Video timestamp in seconds
    outcome = Column(String, default="completed") # completed, intercepted, unsuccessful, uncertain
    confidence = Column(Float, default=1.0)

    job = relationship("AnalysisJob", back_populates="passes")
    options = relationship("PassingOption", back_populates="pass_event", cascade="all, delete-orphan")

class PassingOption(Base):
    __tablename__ = "passing_options"

    id = Column(Integer, primary_key=True, index=True)
    pass_event_id = Column(Integer, ForeignKey("pass_events.id"), nullable=False)
    candidate_track_id = Column(Integer, nullable=False)
    source = Column(String, default="observed")   # observed, temporally_inferred
    score = Column(Float, default=0.0)            # Normalized 0.0 - 1.0
    confidence = Column(Float, default=1.0)
    explanation = Column(Text, nullable=True)

    pass_event = relationship("PassEvent", back_populates="options")

class MissedOpportunity(Base):
    __tablename__ = "missed_opportunities"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    timestamp = Column(Float, nullable=False)
    carrier_track_id = Column(Integer, nullable=False)
    recommended_track_id = Column(Integer, nullable=False)
    score = Column(Float, default=0.0)
    confidence = Column(Float, default=1.0)
    explanation = Column(Text, nullable=True)

    job = relationship("AnalysisJob", back_populates="missed_opportunities")

class PlayerDetection(Base):
    __tablename__ = "player_detections"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    frame_index = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=True)
    x_min = Column(Float, nullable=False)
    y_min = Column(Float, nullable=False)
    x_max = Column(Float, nullable=False)
    y_max = Column(Float, nullable=False)
    center_x = Column(Float, nullable=False)
    center_y = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    class_id = Column(Integer, default=0)

    job = relationship("AnalysisJob", back_populates="detections")
