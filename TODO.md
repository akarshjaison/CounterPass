# CounterPass TODO

## Phase 1: Foundation (Completed)
- [x] Initialize Python FastAPI backend
- [x] Create SQLite database and SQLAlchemy schemas
- [x] Build upload service with video metadata extraction
- [x] Develop full background simulation for Demo Mode
- [x] Build Vite + React frontend dashboard shell
- [x] Implement event details panel and player statistics components
- [x] Connect frontend to backend and run manual + automated tests

## Phase 2: Player Detection (Completed)
- [x] Add YOLO detection pipeline stub/integration
- [x] Save frame-by-frame player bounding boxes and centers

## Phase 3: Multi-Object Tracking (Completed)
- [x] Track players with persistent IDs using ByteTrack/BoT-SORT style logic
- [x] Handle lost-track position estimation and reassociation

## Phase 4: Temporal Buffer (Completed)
- [x] Implement `TemporalFrameBuffer` holding a window of recent histories

## Phase 5: Team Classification (Completed)
- [x] Classify jerseys via color features and K-Means clustering

## Phase 6: Ball Detection and Tracking (Completed)
- [x] Detect and track the ball using temporal smoothing

## Phase 7: Possession Estimation (Completed)
- [x] Predict likely ball carrier based on ball proximity and trajectories

## Phase 8: Pass Event Detection (Completed)
- [x] Detect pass events (completed, incomplete, intercepted) from physics signals

## Phase 9: Passing Option Analysis (Completed)
- [x] Compute distance, pressure, progressions, and options score

## Phase 10: Passing Lane Analysis (Completed)
- [x] Compute lane clear score and opponent intersections

## Phase 11: Missed Opportunities & Decision Metrics (Completed)
- [x] Detect missed options and compute the overall CounterPass Score

## Phase 12: Visual Video Annotator & Final Polish (Completed)
- [x] Render annotated output video (bounding boxes, trajectories, passing options)
