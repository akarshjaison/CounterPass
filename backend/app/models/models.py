from typing import List, Optional
from sqlalchemy import Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime, timezone
from app.db.database import Base

class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    jobs: Mapped[List["AnalysisJob"]] = relationship("AnalysisJob", back_populates="video", cascade="all, delete-orphan")

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    status: Mapped[Optional[str]] = mapped_column(String, default="uploaded")
    progress: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    current_stage: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mode: Mapped[Optional[str]] = mapped_column(String, default="real")
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    video: Mapped["Video"] = relationship("Video", back_populates="jobs")
    tracks: Mapped[List["PlayerTrack"]] = relationship("PlayerTrack", back_populates="job", cascade="all, delete-orphan")
    passes: Mapped[List["PassEvent"]] = relationship("PassEvent", back_populates="job", cascade="all, delete-orphan")
    missed_opportunities: Mapped[List["MissedOpportunity"]] = relationship("MissedOpportunity", back_populates="job", cascade="all, delete-orphan")
    detections: Mapped[List["PlayerDetection"]] = relationship("PlayerDetection", back_populates="job", cascade="all, delete-orphan")

class PlayerTrack(Base):
    __tablename__ = "player_tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    team: Mapped[Optional[str]] = mapped_column(String, default="Unknown")
    confidence: Mapped[Optional[float]] = mapped_column(Float, default=1.0)

    job: Mapped["AnalysisJob"] = relationship("AnalysisJob", back_populates="tracks")

class PassEvent(Base):
    __tablename__ = "pass_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    passer_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    receiver_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    outcome: Mapped[Optional[str]] = mapped_column(String, default="completed")
    confidence: Mapped[Optional[float]] = mapped_column(Float, default=1.0)

    job: Mapped["AnalysisJob"] = relationship("AnalysisJob", back_populates="passes")
    options: Mapped[List["PassingOption"]] = relationship("PassingOption", back_populates="pass_event", cascade="all, delete-orphan")

class PassingOption(Base):
    __tablename__ = "passing_options"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    pass_event_id: Mapped[int] = mapped_column(Integer, ForeignKey("pass_events.id"), nullable=False)
    candidate_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String, default="observed")
    score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    confidence: Mapped[Optional[float]] = mapped_column(Float, default=1.0)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    pass_event: Mapped["PassEvent"] = relationship("PassEvent", back_populates="options")

class MissedOpportunity(Base):
    __tablename__ = "missed_opportunities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    carrier_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    recommended_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    confidence: Mapped[Optional[float]] = mapped_column(Float, default=1.0)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    job: Mapped["AnalysisJob"] = relationship("AnalysisJob", back_populates="missed_opportunities")

class PlayerDetection(Base):
    __tablename__ = "player_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    x_min: Mapped[float] = mapped_column(Float, nullable=False)
    y_min: Mapped[float] = mapped_column(Float, nullable=False)
    x_max: Mapped[float] = mapped_column(Float, nullable=False)
    y_max: Mapped[float] = mapped_column(Float, nullable=False)
    center_x: Mapped[float] = mapped_column(Float, nullable=False)
    center_y: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    class_id: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    job: Mapped["AnalysisJob"] = relationship("AnalysisJob", back_populates="detections")
