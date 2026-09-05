# Prompt for Antigravity IDE

Paste everything below into Antigravity as one task.

---

You are working on the CounterPass repo (FastAPI backend + React frontend, OpenCV/YOLOv8 football analysis pipeline). Go through the codebase and fix the following issues. Work through them in the order listed, run the existing test suite after each group, and add new tests where noted. Do not remove existing passing tests unless they test behavior you're intentionally changing (e.g. the synthetic fallback tests).

## Group 1 — Remove fabricated/fake data (do this first, it's the most damaging issue)

1. In `backend/app/services/analytics_engine.py`, function `run_tactical_analysis`: delete the `if not pass_events:` synthetic fallback block that generates fake passes at fixed timeline fractions (0.15/0.38/0.62/0.85). If no real pass events are detected, the function should return/leave an empty result — do not invent data. Update `compile_match_metrics` to handle the zero-passes case gracefully (already partially does via `total_p > 0` checks — verify it doesn't produce misleading non-zero scores when there's no real data).
2. In `backend/app/services/analysis.py`, function `_ensure_player_tracks`: delete the hardcoded fallback (`[(i, 0.85) for i in range(1,6)] + [(i, 0.80) for i in range(12,17)]`). If no real tracks exist after detection, mark the `AnalysisJob` as `status="failed"` with `error_message="No players detected — check video quality/framing"` instead of silently continuing with fake players.
3. Add an API-level indicator: `GET /analysis/{job_id}/results` and `/events` should surface (in the response or a warning field) when results are based on very few real detections/passes, so the frontend can show a "low confidence" banner instead of presenting sparse/fake-adjacent results as authoritative.

## Group 2 — Fix the passing-option scoring bug

In `backend/app/services/analytics_engine.py`, inside the per-candidate scoring loop (`# F. Composite Option Score`):

1. `dist_score` is currently computed but never used in `composite_score`. Add it as a real weighted term.
2. `pressure_score` is currently multiplied by both `WEIGHT_SPACE_SCORE` and `WEIGHT_PRESSURE_RISK` (same value, two weights — double counting). Same problem with `lane_clearance` under `WEIGHT_LANE_CLEARANCE` and `WEIGHT_INTERCEPTION_RISK`.
3. Fix this by either:
   - (a) Adding `dist_score` as a genuine 7th weighted term and rebalancing `core/config.py` weights so they still sum to 1.0, or
   - (b) Replacing the currently-duplicated `WEIGHT_INTERCEPTION_RISK` term with a real interception-risk calculation: probability that the *nearest opponent* can intercept the pass, based on opponent distance-to-lane vs opponent's average speed (from `buffer.get_average_velocity`) vs pass flight time — not a copy of `lane_clearance`.
4. Update `backend/tests/test_analytics.py` to assert that distance meaningfully affects `composite_score` (e.g. two otherwise-identical candidates at different distances should get different scores).

## Group 3 — Fix broken metrics

In `backend/app/services/analytics_engine.py`, `compile_match_metrics`:

1. `forward_passes` currently uses `if p.receiver_track_id > p.passer_track_id`. Replace with a real geometric check: look up passer/receiver x-position at the pass frame and compare against that team's attack direction (see Group 4 for how attack direction should be determined).
2. `movement_rating` is hardcoded to `80.0`. Replace with a real aggregate: average speed (from `buffer.get_average_velocity`) across all of a team's/player's tracked frames in the job, normalized to a 0–100 scale with a sensible reference max speed.

## Group 4 — Pitch calibration (this is the biggest accuracy lever after Group 1)

Currently there is no homography/pitch mapping despite the README describing one — all distance/lane/possession thresholds operate in raw video pixels, which breaks whenever camera zoom/distance/framing differs.

1. Add a calibration step: either
   - a simple one-time UI step where the user clicks the 4 corners of the pitch (or penalty box) on the first frame of the video, or
   - automatic pitch-line detection via Hough transform on the white markings, with manual override available.
2. Compute a homography matrix from the 4 correspondences (real-world pitch dimensions are standard: 105m x 68m, configurable) and store it per `AnalysisJob`.
3. Before computing any distance-based score (`dist_score`, `lane_clearance`, `POSSESSION_DISTANCE_THRESHOLD`, pass distance checks), transform player/ball pixel coordinates through the homography into meters. Convert existing pixel-based constants in `core/config.py` into meter-based equivalents.
4. Determine attack direction dynamically per team: compute each team's average x-position (in pitch coordinates) over the first ~30 seconds of tracked data, and infer which goal they're attacking. Re-check this once past the midpoint of the video duration in case of a half-time end-swap.
5. Add tests covering the homography transform (round-trip a few known points) and the attack-direction inference on a synthetic dataset.

## Group 5 — Detection quality

1. Replace or supplement the raw COCO-pretrained `yolov8n.pt` with a football-fine-tuned model if feasible (note in code/README that this is a TODO if you can't source a fine-tuned checkpoint in this pass — don't fabricate one).
2. Raise the ball-detection confidence threshold in `detection.py` from `0.1` to something in the `0.25–0.3` range to cut false positives, and make it a named constant in `core/config.py` rather than a magic number.
3. Add appearance-based re-identification to `SimpleTracker` (a simple color/histogram feature vector per track, compared alongside IoU/distance) to reduce ID switches after occlusion, which currently degrades team mapping stability.

## Group 6 — Ground-truth eval harness

1. Add a `backend/eval/` directory with a script that runs the full pipeline against 2-3 short manually-labeled clips (pass timestamps + passer/receiver identity + ball position sampled every N frames — you'll need to help me create this labeled data or point me to how to produce it, don't invent ground truth).
2. Script should report: pass-detection precision/recall, mean ball/player position error (once calibration exists, report in meters), and team-classification accuracy.
3. Wire this into CI or at least a documented `make eval` / script command so accuracy changes can be measured before/after future changes.

## Group 7 — Performance (loading/processing is slow)

1. Add indexes: `job_id` on `PlayerDetection`, `PassEvent`, `PlayerTrack`, `MissedOpportunity`, `PassingOption` (via `pass_event_id`), and `track_id` on `PlayerDetection`, in `backend/app/models/models.py`. Generate/apply the corresponding migration.
2. In `detection.py`, replace the O(n×m) pattern of filtering the full `detections` list per track ID (`[d for d in detections if d.track_id == tid]`, appears twice) with a single pre-grouped dict keyed by `track_id`, built in one pass.
3. Stop calling `sanitize_job_data(db, job_id)` on every `GET /results` and `GET /events` request. Run it once, right after `run_tactical_analysis` completes in the background worker (`analysis.py`), and make the GET endpoints read-only.
4. Investigate replacing `cv2.dnn.readNetFromONNX(...).forward()` per-frame CPU inference with either: `onnxruntime` with `CUDAExecutionProvider` if a GPU is available (fall back to CPU otherwise), or batched inference (accumulate N frames, run one batched forward pass) to reduce per-call overhead. Benchmark before/after on a sample video and report the numbers.
5. Check the frontend polling interval for job status (`frontend/src/App.jsx` or wherever it polls) — if it's polling faster than ~1-2s, back it off; consider replacing polling with a WebSocket or SSE push from the backend for progress updates instead.
6. After each change in this group, benchmark end-to-end processing time on the same test video and report before/after numbers in the PR description.

## General instructions

- Make changes incrementally, one group at a time, and run `pytest backend/tests/` after each group.
- Don't silently swallow exceptions anywhere you touch — if you find more `except Exception: pass`-style fallbacks like the ones in Group 1, flag them to me rather than assuming they should be removed.
- Update `README.md` / `CHANGELOG.md` to reflect what changed, especially the removal of synthetic fallback data (this affects how results should be interpreted).
- If any fix requires a data/model asset I don't have (e.g. a fine-tuned YOLO checkpoint, labeled eval clips), stop and ask me rather than fabricating placeholder data.
