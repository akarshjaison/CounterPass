# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-08-12 - Core Refactor & System Fixes (Groups 1-7)
### Changed & Fixed
- **Fabricated Data Removal**: Removed all synthetic fallback pass events and hardcoded player track fallbacks. Jobs with 0 player detections fail gracefully with `"No players detected"` error status.
- **Scoring & Interception Risk**: Added `dist_score` to composite option score formula, rebalanced weights to sum to 1.0, and replaced duplicate terms with a real physics-based interception safety calculation.
- **Geometric Metrics**: Replaced arbitrary track_id comparisons in `forward_passes` with geometric x-position checks relative to team attack directions. Replaced hardcoded `movement_rating` with calculated average speed across tracked frames.
- **Pitch Calibration**: Added `calibration.py` for 4-point homography matrix calculation ($105\text{m} \times 68\text{m}$ pitch space) and dynamic team attack direction inference.
- **Detection Quality**: Raised ball detection confidence threshold to `0.25` via named config constant and added pre-grouped dictionary lookups in detection tracking.
- **Ground-Truth Eval Harness**: Created `backend/eval/eval_harness.py` for precision/recall, position error (MAE/RMSE), and team accuracy evaluation against ground-truth datasets.
- **Performance & Read-Only Endpoints**: Added database indexes on `job_id`, `track_id`, and `pass_event_id` in `models.py`. Removed data mutation calls from `GET` endpoints to make them read-only.

## [1.1.0] - 2026-07-11 - Phase 2: Player Detection
### Added
- Database model schema for frame-by-frame `PlayerDetection` (storing bounding boxes, centers, and confidences).
- Integrated YOLOv8 object detection pipeline using OpenCV's DNN module.
- High-fidelity simulated player/ball trajectory generator that degrades gracefully when YOLO weights are missing.
- REST API endpoint `/api/v1/analysis/{job_id}/detections` to query sorted frame detections.
- Endpoint and end-to-end integration tests using pytest.

## [1.0.0] - 2026-07-11 - Phase 1: Foundation
### Added
- Python FastAPI backend initialization with SQLite database and SQLAlchemy schemas.
- Video upload service with frame metadata extraction (duration, FPS, dimensions).
- Background threaded simulation worker for Demo Mode populating detailed mockup analytics.
- Vite + React frontend dashboard featuring:
  - Match Overview with CounterPass score, completion rates, and missed lanes.
  - Interactive SVG pitch mapping passing lane options (observed & temporally inferred) and opponent locations.
  - Pass decision detail panel and player squad list with spatial awareness/decision metrics.
- Automated API test suite using pytest.

