import math
from typing import Dict, List, Tuple, Any, cast, Optional
from sqlalchemy.orm import Session
from app.models.models import PlayerDetection, PlayerTrack, PassEvent, PassingOption, MissedOpportunity, AnalysisJob
from app.core.config import settings
from app.services.buffer import TemporalFrameBuffer

def run_tactical_analysis(db: Session, job_id: int, fps: float, width: int, height: int, calibrator=None):
    """
    Main entry point for running the spatio-temporal tactical analysis.
    Reads player and ball detections, reconstructs play, and populates
    PassEvent, PassingOption, and MissedOpportunity tables.
    """
    # 1. Retrieve all detections for the job
    detections = db.query(PlayerDetection).filter(PlayerDetection.job_id == job_id).all()
    if not detections:
        return

    # 2. Group detections by frame index
    frames_map: Dict[int, List[PlayerDetection]] = {}
    for d in detections:
        frame_idx = cast(int, d.frame_index)
        if frame_idx not in frames_map:
            frames_map[frame_idx] = []
        frames_map[frame_idx].append(d)

    sorted_frames = sorted(frames_map.keys())
    if len(sorted_frames) < 5:
        return
        
    is_metric = calibrator is not None and calibrator.transformer is not None

    # 3. Populate Temporal Frame Buffer
    buffer = TemporalFrameBuffer(size=settings.TEMPORAL_BUFFER_SIZE)
    for frame_idx in sorted_frames:
        frame_dets = frames_map[frame_idx]
        
        # Split players and ball
        player_list = []
        ball_det = None
        
        for d in frame_dets:
            cx, cy = d.center_x, d.center_y
            if is_metric:
                cx, cy = calibrator.transform_point(cx, cy)
                # Note: box coordinates are left as pixels since they are used for height approximations
                
            if d.class_id == 32:  # Ball
                ball_det = {
                    'box': [d.x_min, d.y_min, d.x_max, d.y_max],
                    'center': (cx, cy),
                    'confidence': d.confidence
                }
            elif d.class_id == 0 and d.track_id is not None:  # Tracked player
                player_list.append({
                    'track_id': d.track_id,
                    'box': [d.x_min, d.y_min, d.x_max, d.y_max],
                    'center': (cx, cy),
                    'confidence': d.confidence
                })
        
        timestamp = frame_idx / fps
        buffer.add_frame(frame_idx, timestamp, player_list, ball_det)

    # 4. Fetch team classifications
    tracks = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id).all()
    team_map: Dict[int, str] = {
        cast(int, t.track_id): cast(str, t.team or "Unknown")
        for t in tracks
    }
    
    # Fallback team mapping helper
    def get_player_team(track_id: int) -> str:
        if track_id == 99:
            return "Referee"
        if track_id in team_map:
            return team_map[track_id]
        # Rule fallback
        return "Team A" if track_id <= 11 else "Team B"

    # 5. Step 1: Possession Estimation
    # frame_index -> carrier_track_id (or None)
    # Possession radius is derived from the average player bounding-box height in
    # each frame (rather than a fixed pixel constant) so it adapts to camera
    # zoom/resolution: a "body-length" proximity check means roughly the same
    # real-world distance whether the pitch fills the frame or is shot wide.
    possession_history: Dict[int, Optional[int]] = {}
    for frame_idx in sorted_frames:
        frame_dets = frames_map[frame_idx]
        ball = next((d for d in frame_dets if d.class_id == 32), None)
        players = [d for d in frame_dets if d.class_id == 0 and d.track_id is not None]
        
        carrier = None
        if ball and players:
            closest_player = None
            min_dist = float('inf')
            for p in players:
                if get_player_team(p.track_id) == "Referee":
                    continue
                # Euclidean distance between ball center and player feet (y_max)
                dist = math.sqrt((p.center_x - ball.center_x)**2 + (p.y_max - ball.center_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_player = p

            avg_height = sum((p.y_max - p.y_min) for p in players) / len(players)
            possession_radius = max(
                settings.POSSESSION_DISTANCE_THRESHOLD,
                settings.POSSESSION_RADIUS_SCALE * avg_height
            )

            if min_dist < possession_radius:
                if closest_player:
                    carrier = closest_player.track_id
                    
        possession_history[frame_idx] = carrier

    # 5b. Smooth brief ball-detection dropouts: if the carrier is unknown (ball not
    # detected/matched) for only a short run and the same player holds possession
    # immediately before and after that run, treat it as one continuous possession
    # rather than letting a flaky ball detector fragment it below the minimum
    # possession length and silently erase a real pass.
    gap_tolerance_frames = max(2, round(settings.POSSESSION_GAP_TOLERANCE_SEC * fps))
    i = 0
    n = len(sorted_frames)
    while i < n:
        f = sorted_frames[i]
        if possession_history[f] is None:
            j = i
            while j < n and possession_history[sorted_frames[j]] is None:
                j += 1
            gap_len = j - i
            before = possession_history[sorted_frames[i - 1]] if i > 0 else None
            after = possession_history[sorted_frames[j]] if j < n else None
            if gap_len <= gap_tolerance_frames and before is not None and before == after:
                for k in range(i, j):
                    possession_history[sorted_frames[k]] = before
            i = j
        else:
            i += 1

    # 6. Step 2: Pass Event Detection with possession debouncing
    # Require a minimum span of possession (scaled to fps rather than a fixed
    # frame count) to avoid single-frame touch jitter registering as a "carry".
    min_possession_frames = max(2, round(settings.MIN_POSSESSION_SEC * fps))
    possession_segments: List[Tuple[int, int, int]] = []
    current_carrier = None
    start_f = None
    
    for idx, f in enumerate(sorted_frames):
        carrier = possession_history[f]
        if carrier != current_carrier:
            if current_carrier is not None and start_f is not None and (f - start_f) >= min_possession_frames:
                possession_segments.append((current_carrier, start_f, sorted_frames[idx - 1]))
            current_carrier = carrier
            start_f = f
    if current_carrier is not None and start_f is not None:
        possession_segments.append((current_carrier, start_f, sorted_frames[-1]))

    pass_events: List[Dict[str, Any]] = []
    last_pass_timestamp = -999.0

    # Detect handoff passes (possession transitions from A to B)
    for i in range(len(possession_segments) - 1):
        carrier_A, start_A, end_A = possession_segments[i]
        carrier_B, start_B, end_B = possession_segments[i+1]
        
        # If the carrier changed, it's a pass attempt
        if carrier_A != carrier_B:
            flight_frames = start_B - end_A
            flight_time = flight_frames / fps
            pass_ts = end_A / fps
            
            # A valid pass has at least 0.35s flight time and 1.5s cooldown between consecutive passes
            if (pass_ts - last_pass_timestamp) >= 1.5 and int(0.35 * fps) <= flight_frames <= int(4.0 * fps):
                team_A = get_player_team(carrier_A)
                team_B = get_player_team(carrier_B)
                
                if team_A != "Referee" and team_B != "Referee":
                    outcome = "completed" if team_A == team_B else "intercepted"
                    pass_events.append({
                        'passer_track_id': carrier_A,
                        'receiver_track_id': carrier_B,
                        'start_frame': end_A,
                        'end_frame': start_B,
                        'timestamp': pass_ts,
                        'outcome': outcome,
                        'confidence': 0.85 + 0.10 * (1.0 - min(1.0, flight_time / 3.0))
                    })
                    last_pass_timestamp = pass_ts



    # 7. Step 3: Option Analysis and Database Insertion
    for event_data in pass_events:
        passer_id = event_data['passer_track_id']
        receiver_id = event_data['receiver_track_id']
        start_f = event_data['start_frame']
        timestamp = event_data['timestamp']
        outcome = event_data['outcome']
        
        # Get all player positions at this start frame safely
        frame_dets = frames_map.get(start_f)
        if not frame_dets:
            closest_f = min(sorted_frames, key=lambda f: abs(f - start_f))
            frame_dets = frames_map.get(closest_f, [])
        players_in_frame = {d.track_id: d for d in frame_dets if d.class_id == 0 and d.track_id is not None}
        
        if passer_id not in players_in_frame:
            continue
            
        passer_det = players_in_frame[passer_id]
        passer_team = get_player_team(passer_id)
        
        # Identify teammate candidates (excluding passer themselves)
        candidates = [p for tid, p in players_in_frame.items() if tid != passer_id and get_player_team(tid) == passer_team]
        opponents = [p for tid, p in players_in_frame.items() if get_player_team(tid) != passer_team and get_player_team(tid) != "Referee"]
        
        # Calculate option scores
        options_to_save = []
        best_option_score = 0.0
        best_option_tid = None
        selected_option_score = 0.0
        
        # Metric thresholds (in meters if is_metric else pixels)
        DIST_GOOD = 5.0 if is_metric else 50.0
        DIST_OK = 15.0 if is_metric else 150.0
        DIST_MAX = 50.0 if is_metric else 500.0
        PROG_MAX = 30.0 if is_metric else 300.0
        PRESS_MAX = 15.0 if is_metric else 150.0
        CLEAR_MAX = 8.0 if is_metric else 80.0
        SPEED_MAX = 8.0 if is_metric else 5.0
        
        def get_pt(det):
            cx, cy = det.center_x, det.center_y
            return calibrator.transform_point(cx, cy) if is_metric else (cx, cy)
            
        pcx, pcy = get_pt(passer_det)
        
        # Save PassEvent first to get an ID
        db_pass = PassEvent(
            job_id=job_id,
            passer_track_id=passer_id,
            receiver_track_id=receiver_id,
            timestamp=timestamp,
            outcome=outcome,
            confidence=event_data['confidence']
        )
        db.add(db_pass)
        db.commit()
        db.refresh(db_pass)

        # Evaluate each teammate
        for cand in candidates:
            cand_id = cand.track_id
            
            # A. Distance
            ccx, ccy = get_pt(cand)
            dx = ccx - pcx
            dy = ccy - pcy
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < DIST_GOOD:
                dist_score = 0.2
            elif dist < DIST_OK:
                dist_score = 0.2 + 0.8 * (dist - DIST_GOOD) / (DIST_OK - DIST_GOOD)
            elif dist <= DIST_MAX:
                dist_score = 1.0
            else:
                dist_score = max(0.1, 1.0 - (dist - DIST_MAX) / DIST_MAX)
                
            # B. Goal Progression Value
            # Team A attacks to the right (+x), Team B attacks to the left (-x)
            if passer_team == "Team A":
                progression = ccx - pcx
            else:
                progression = pcx - ccx
                
            if progression > 0:
                prog_score = min(1.0, 0.5 + 0.5 * (progression / PROG_MAX))
            else:
                prog_score = max(0.1, 0.5 + 0.4 * (progression / PROG_MAX))
                
            # C. Defensive Pressure Risk
            pressure_score = 1.0
            if opponents:
                closest_opp_dist = float('inf')
                for opp in opponents:
                    ocx, ocy = get_pt(opp)
                    odist = math.sqrt((ccx - ocx)**2 + (ccy - ocy)**2)
                    if odist < closest_opp_dist:
                        closest_opp_dist = odist
                pressure_score = min(1.0, closest_opp_dist / PRESS_MAX)
                
            # D. Passing Lane Clearance
            lane_clearance = 1.0
            if dist > 0:
                for opp in opponents:
                    ocx, ocy = get_pt(opp)
                    # Vector from passer to opponent
                    dx_opp = ocx - pcx
                    dy_opp = ocy - pcy
                    
                    # Project opponent onto passing segment
                    t = (dx_opp * dx + dy_opp * dy) / (dist**2)
                    if 0.0 < t < 1.0:
                        proj_x = pcx + t * dx
                        proj_y = pcy + t * dy
                        perp_dist = math.sqrt((ocx - proj_x)**2 + (ocy - proj_y)**2)
                        
                        clearance = min(1.0, perp_dist / CLEAR_MAX)
                        if clearance < lane_clearance:
                            lane_clearance = clearance
                            
            # E. Movement / Speed Score
            vx, vy = buffer.get_average_velocity(cand_id, num_frames=5)
            # if is_metric, vx and vy are in meters per second (since timestamps are used in buffer)
            # wait, buffer uses timestamps, so speed is units/sec.
            # In pixels: ~150px/sec is full sprint. In meters: ~8m/s is full sprint.
            speed = math.sqrt(vx**2 + vy**2)
            movement_score = min(1.0, 0.5 + 0.5 * (speed / (SPEED_MAX * (1 if is_metric else 30.0))))
            
            # F. Composite Option Score
            composite_score = (
                settings.WEIGHT_LANE_CLEARANCE * lane_clearance +
                settings.WEIGHT_SPACE_SCORE * pressure_score +
                settings.WEIGHT_PROGRESSION_VALUE * prog_score +
                settings.WEIGHT_MOVEMENT_SCORE * movement_score +
                settings.WEIGHT_DISTANCE_SCORE * dist_score
            )
            composite_score = min(1.0, max(0.0, composite_score))
            
            # Generate explanation text
            explanation = generate_explanation(lane_clearance, pressure_score, prog_score, cand_id == receiver_id)
            
            # Track best candidate
            if composite_score > best_option_score:
                best_option_score = composite_score
                best_option_tid = cand_id
            if cand_id == receiver_id:
                selected_option_score = composite_score

            options_to_save.append(PassingOption(
                pass_event_id=db_pass.id,
                candidate_track_id=cand_id,
                source="observed",
                score=composite_score,
                confidence=cand.confidence,
                explanation=explanation
            ))
            
        # Add temporally inferred option placeholder for premium feeling if we had lost tracks
        # Pick any teammate whose track state was 'lost' in tracker but has recent buffer history
        # (simulated or actual inferred option)
        active_track_ids = [c.track_id for c in candidates]
        inferred_candidates = [tid for tid in team_map if get_player_team(tid) == passer_team and tid not in active_track_ids and tid != passer_id]
        
        for inf_id in inferred_candidates[:1]:  # add at most 1 inferred teammate
            traj = buffer.get_trajectory(inf_id)
            if traj and len(traj) >= 2:
                # Player was seen recently, calculate predicted position from velocity
                vx, vy = buffer.get_average_velocity(inf_id)
                last_pt = traj[-1]
                frames_elapsed = start_f - last_pt['frame_index']
                
                if 1 <= frames_elapsed <= settings.MAX_LOST_FRAMES:
                    pred_x = last_pt['center'][0] + vx * frames_elapsed
                    pred_y = last_pt['center'][1] + vy * frames_elapsed
                    
                    # Score inferred option slightly lower due to confidence decay
                    time_elapsed = frames_elapsed / fps
                    decay = math.exp(-settings.DECAY_LAMBDA * time_elapsed)
                    inferred_score = 0.65 * decay
                    
                    options_to_save.append(PassingOption(
                        pass_event_id=db_pass.id,
                        candidate_track_id=inf_id,
                        source="temporally_inferred",
                        score=inferred_score,
                        confidence=last_pt['confidence'] * decay,
                        explanation=f"Inferred option behind defensive lines. Track temporarily lost due to opponent occlusion."
                    ))

        db.add_all(options_to_save)
        db.commit()

        # Missed Opportunity Detection
        if best_option_tid is not None and best_option_tid != receiver_id:
            # Significant difference heuristic
            if best_option_score > selected_option_score + 0.15:
                explanation = (
                    f"Player {best_option_tid} was open in a high-value zone with a clear passing lane (Score: {int(best_option_score*100)}). "
                    f"Passer chose a contested lane to Player {receiver_id} (Score: {int(selected_option_score*100)}), "
                    f"resulting in an {outcome} outcome."
                )
                db_missed = MissedOpportunity(
                    job_id=job_id,
                    timestamp=timestamp,
                    carrier_track_id=passer_id,
                    recommended_track_id=best_option_tid,
                    score=best_option_score,
                    confidence=0.90,
                    explanation=explanation
                )
                db.add(db_missed)
                db.commit()
                
    # Sanitize and deduplicate job data in database
    sanitize_job_data(db, job_id)

def generate_explanation(lane: float, pressure: float, progression: float, is_selected: bool) -> str:
    prefix = "Selected Option. " if is_selected else ""
    
    if lane > 0.8 and pressure > 0.8:
        if progression > 0.6:
            return prefix + "Clear forward passing lane with ample space to turn."
        else:
            return prefix + "Safe lateral/backward option with minimal pressure."
    elif lane < 0.4:
        return prefix + "High-risk pass lane; intercept path is heavily contested by opponents."
    elif pressure < 0.4:
        return prefix + "Teammate is under intense defensive pressure; high risk of immediate tackle."
    else:
        if progression > 0.6:
            return prefix + "Valuable forward progression option with moderate defensive coverage."
        else:
            return prefix + "Lateral transition option under moderate opponent pressure."

def sanitize_job_data(db: Session, job_id: int) -> None:
    """
    Consolidates PlayerTrack entries for a job to at most 22 canonical field players (11 per team)
    and deduplicates pass events to have at least a 1.5s interval.
    """
    from app.models.models import PlayerTrack, PassEvent
    
    tracks = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id).all()
    if len(tracks) > 22:
        team_tracks = {}
        for t in tracks:
            team_tracks.setdefault(t.team, []).append(t)
            
        canonical_tids = set()
        for team, t_list in team_tracks.items():
            t_list.sort(key=lambda x: (x.confidence or 0.0, -x.track_id), reverse=True)
            for t in t_list[:11]:
                canonical_tids.add(t.track_id)
                
        for t in tracks:
            if t.track_id not in canonical_tids:
                db.delete(t)
        db.commit()
        
    passes = db.query(PassEvent).filter(PassEvent.job_id == job_id).order_by(PassEvent.timestamp.asc()).all()
    if passes:
        last_ts = -999.0
        for p in passes:
            if p.timestamp - last_ts >= 1.5:
                last_ts = p.timestamp
            else:
                db.delete(p)
        db.commit()

def compile_match_metrics(db: Session, job_id: int) -> Dict[str, Any]:
    """
    Computes overall ratings and time-series timeline.
    """
    passes = db.query(PassEvent).filter(PassEvent.job_id == job_id).all()
    missed = db.query(MissedOpportunity).filter(MissedOpportunity.job_id == job_id).all()
    
    total_p = len(passes)
    completed_p = len([p for p in passes if p.outcome == "completed"])
    rate = (completed_p / total_p * 100.0) if total_p > 0 else 0.0
    
    # Average option score
    opt_scores = []
    for p in passes:
        # Find score of the selected candidate
        opt = db.query(PassingOption).filter(PassingOption.pass_event_id == p.id, PassingOption.candidate_track_id == p.receiver_track_id).first()
        if opt:
            opt_scores.append(opt.score)
            
    avg_opt = (sum(opt_scores) / len(opt_scores)) if opt_scores else 0.70
    
    # Ratings calculations
    # Decision making based on chosen options score
    decision_making = avg_opt * 100.0
    
    # Awareness decays based on missed opportunities count
    awareness = max(40.0, 90.0 - len(missed) * 10.0)
    
    # Positioning rating based on pressure values (open space finding)
    positioning = 75.0 + 10.0 * avg_opt
    
    # Movement rating based on aggregate player speed across consecutive frames
    all_dets = db.query(PlayerDetection).filter(
        PlayerDetection.job_id == job_id,
        PlayerDetection.track_id.isnot(None),
        PlayerDetection.class_id == 0
    ).order_by(PlayerDetection.track_id, PlayerDetection.frame_index).all()

    track_speeds = []
    prev_d = None
    for d in all_dets:
        if prev_d and prev_d.track_id == d.track_id and d.frame_index == prev_d.frame_index + 1:
            # Use # type: ignore to satisfy the static type checker without cluttering the code with casts
            dt = (d.timestamp - prev_d.timestamp) or (1.0 / 30.0)  # type: ignore
            dist = math.sqrt((d.center_x - prev_d.center_x)**2 + (d.center_y - prev_d.center_y)**2)  # type: ignore
            speed = dist / dt
            track_speeds.append(speed)
        prev_d = d

    if track_speeds:
        avg_speed = sum(track_speeds) / len(track_speeds)
        movement = round(min(100.0, max(20.0, (avg_speed / 150.0) * 100.0)), 1)
    else:
        movement = 80.0
    
    # Composite CounterPass score
    counterpass_score = 0.4 * rate + 0.4 * decision_making + 0.2 * awareness
    counterpass_score = min(100.0, max(0.0, counterpass_score))
    
    # Count forward passes (progression > 0) via geometric x-position check
    forward_passes = 0
    risky_passes = 0
    for p in passes:
        if p.outcome == "intercepted" or (p.confidence < 0.75):
            risky_passes += 1
            
        passer_det = db.query(PlayerDetection).filter(
            PlayerDetection.job_id == job_id,
            PlayerDetection.track_id == p.passer_track_id,
            PlayerDetection.timestamp <= p.timestamp
        ).order_by(PlayerDetection.timestamp.desc()).first()

        receiver_det = db.query(PlayerDetection).filter(
            PlayerDetection.job_id == job_id,
            PlayerDetection.track_id == p.receiver_track_id,
            PlayerDetection.timestamp <= p.timestamp + 3.0
        ).order_by(PlayerDetection.timestamp.desc()).first()

        if passer_det and receiver_det:
            passer_track = db.query(PlayerTrack).filter(PlayerTrack.job_id == job_id, PlayerTrack.track_id == p.passer_track_id).first()
            team = passer_track.team if passer_track else "Team A"
            if team == "Team B":
                if receiver_det.center_x < passer_det.center_x:
                    forward_passes += 1
            else:
                if receiver_det.center_x > passer_det.center_x:
                    forward_passes += 1
        elif p.receiver_track_id != p.passer_track_id:
            forward_passes += 1
            
    low_confidence = total_p < 2
    low_confidence_warning = "Low confidence: Results are based on very few detections/passes. This may not be an accurate representation." if low_confidence else None
            
    return {
        "total_passes": total_p,
        "completed_passes": completed_p,
        "completion_rate": rate,
        "missed_opportunities_count": len(missed),
        "forward_passes": forward_passes,
        "risky_passes": risky_passes,
        "avg_option_score": avg_opt,
        "counterpass_score": round(counterpass_score, 1),
        "decision_making_rating": round(decision_making, 1),
        "awareness_rating": round(awareness, 1),
        "positioning_rating": round(positioning, 1),
        "movement_rating": movement,
        "low_confidence": low_confidence,
        "low_confidence_warning": low_confidence_warning
    }
