import math

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as [x_min, y_min, x_max, y_max].
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = float(box1_area + box2_area - intersection_area)
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area

class SimpleTracker:
    def __init__(self, max_lost_frames=30):
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks = {}  # track_id -> track_dict

    def update(self, detections):
        """
        Updates the tracker with current frame detections.
        detections: list of dicts: {'box': [xmin, ymin, xmax, ymax], 'confidence': float}
        Returns: list of active track dicts that are currently visible/tracked.
        """
        # 1. Predict position for all active tracks using constant velocity
        for track_id, track in list(self.tracks.items()):
            vx, vy = track['velocity']
            # Apply velocity translation
            track['box'] = [
                track['box'][0] + vx,
                track['box'][1] + vy,
                track['box'][2] + vx,
                track['box'][3] + vy
            ]
            track['center'] = (
                (track['box'][0] + track['box'][2]) / 2.0,
                (track['box'][1] + track['box'][3]) / 2.0
            )

        matched_tracks = {}  # track_id -> detection_idx
        matched_dets = {}    # detection_idx -> track_id

        # 2. First association step: Match using IoU (threshold = 0.3)
        iou_candidates = []
        for track_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                iou = compute_iou(track['box'], det['box'])
                if iou >= 0.3:
                    iou_candidates.append((iou, track_id, det_idx))

        # Greedy match based on highest IoU first
        iou_candidates.sort(key=lambda x: x[0], reverse=True)
        for iou, track_id, det_idx in iou_candidates:
            if track_id not in matched_tracks and det_idx not in matched_dets:
                matched_tracks[track_id] = det_idx
                matched_dets[det_idx] = track_id

        # 3. Second association step: Match remaining unmatched using center-to-center distance
        unmatched_tracks = [tid for tid in self.tracks.keys() if tid not in matched_tracks]
        unmatched_dets = [idx for idx in range(len(detections)) if idx not in matched_dets]

        dist_candidates = []
        for tid in unmatched_tracks:
            track = self.tracks[tid]
            for idx in unmatched_dets:
                det = detections[idx]
                tc = track['center']
                dc = (
                    (det['box'][0] + det['box'][2]) / 2.0,
                    (det['box'][1] + det['box'][3]) / 2.0
                )
                dist = math.sqrt((tc[0] - dc[0])**2 + (tc[1] - dc[1])**2)
                if dist < 220.0:  # Match if centers are within 220 pixels
                    dist_candidates.append((dist, tid, idx))

        # Greedy match based on smallest distance first
        dist_candidates.sort(key=lambda x: x[0])
        for dist, tid, idx in dist_candidates:
            if tid not in matched_tracks and idx not in matched_dets:
                matched_tracks[tid] = idx
                matched_dets[idx] = tid

        # 4. Process matched tracks
        updated_tracks = {}
        for tid, idx in matched_tracks.items():
            track = self.tracks[tid]
            det = detections[idx]
            new_box = det['box']
            new_center = (
                (new_box[0] + new_box[2]) / 2.0,
                (new_box[1] + new_box[3]) / 2.0
            )
            
            # Compute track velocity (displacement)
            old_center = track['center']
            vx = new_center[0] - old_center[0]
            vy = new_center[1] - old_center[1]
            
            # Dampen velocity change if previously tracked
            if track['state'] == 'tracked':
                vx = 0.6 * vx + 0.4 * track['velocity'][0]
                vy = 0.6 * vy + 0.4 * track['velocity'][1]
            
            track['box'] = new_box
            track['center'] = new_center
            track['velocity'] = (vx, vy)
            track['state'] = 'tracked'
            track['lost_count'] = 0
            track['confidence'] = det['confidence']
            updated_tracks[tid] = track

        # 5. Process unmatched tracks (either mark lost or purge)
        for tid in self.tracks.keys():
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track['lost_count'] += 1
                if track['lost_count'] <= self.max_lost_frames:
                    track['state'] = 'lost'
                    # Gradually decay velocity when track is lost
                    track['velocity'] = (track['velocity'][0] * 0.85, track['velocity'][1] * 0.85)
                    updated_tracks[tid] = track

        # 6. Initialize or re-associate tracks for unmatched high-confidence detections
        unmatched_dets = [idx for idx in range(len(detections)) if idx not in matched_dets]
        for idx in unmatched_dets:
            det = detections[idx]
            if det['confidence'] >= 0.4:
                box = det['box']
                center = (
                    (box[0] + box[2]) / 2.0,
                    (box[1] + box[3]) / 2.0
                )
                
                # Check if we can re-associate with any lost track within 250px
                best_lost_tid = None
                best_dist = float('inf')
                for tid, t in updated_tracks.items():
                    if t.get('state') == 'lost':
                        tc = t['center']
                        dist = math.sqrt((center[0] - tc[0])**2 + (center[1] - tc[1])**2)
                        if dist < 250.0 and dist < best_dist:
                            best_dist = dist
                            best_lost_tid = tid
                            
                if best_lost_tid is not None:
                    tid = best_lost_tid
                    updated_tracks[tid] = {
                        'id': tid,
                        'box': box,
                        'center': center,
                        'velocity': (0.0, 0.0),
                        'lost_count': 0,
                        'state': 'tracked',
                        'confidence': det['confidence']
                    }
                elif self.next_id <= 24:
                    tid = self.next_id
                    self.next_id += 1
                    updated_tracks[tid] = {
                        'id': tid,
                        'box': box,
                        'center': center,
                        'velocity': (0.0, 0.0),
                        'lost_count': 0,
                        'state': 'tracked',
                        'confidence': det['confidence']
                    }
                else:
                    # Recycle oldest/closest lost track to prevent ID explosion beyond 24
                    lost_tids = [tid for tid, t in updated_tracks.items() if t.get('state') == 'lost']
                    if lost_tids:
                        closest_tid = min(lost_tids, key=lambda tid: math.sqrt((center[0] - updated_tracks[tid]['center'][0])**2 + (center[1] - updated_tracks[tid]['center'][1])**2))
                        tid = closest_tid
                        updated_tracks[tid] = {
                            'id': tid,
                            'box': box,
                            'center': center,
                            'velocity': (0.0, 0.0),
                            'lost_count': 0,
                            'state': 'tracked',
                            'confidence': det['confidence']
                        }

        self.tracks = updated_tracks
        # Return currently active tracks
        return [t for t in self.tracks.values() if t['state'] == 'tracked']

class BallTracker:
    def __init__(self, max_lost_frames=10):
        self.max_lost_frames = max_lost_frames
        self.last_position = None  # (cx, cy)
        self.last_box = None  # [xmin, ymin, xmax, ymax]
        self.velocity = (0.0, 0.0)  # (vx, vy)
        self.lost_count = 0
        self.state = 'lost'
        self.confidence = 0.0

    def update(self, ball_detections):
        """
        Updates ball track using current frame ball detections.
        ball_detections: list of dicts: {'box': [xmin, ymin, xmax, ymax], 'confidence': float}
        Returns: Dict containing the current tracked or predicted ball state, or None if lost.
        """
        # If lost and no detections, return None
        if self.state == 'lost' and not ball_detections:
            return None

        # Predict current position using velocity if we were tracked
        pred_pos = None
        if self.state == 'tracked' and self.last_position:
            pred_pos = (
                self.last_position[0] + self.velocity[0],
                self.last_position[1] + self.velocity[1]
            )

        matched_det = None
        if ball_detections:
            # Pick the best detection:
            # If we had a predicted position, choose detection closest to prediction
            if pred_pos:
                dists = []
                for det in ball_detections:
                    box = det['box']
                    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                    dist = math.sqrt((center[0] - pred_pos[0])**2 + (center[1] - pred_pos[1])**2)
                    dists.append((dist, det))
                # Sort by distance
                dists.sort(key=lambda x: x[0])
                # Only match if within 150 pixels distance
                if dists[0][0] < 150.0:
                    matched_det = dists[0][1]
            else:
                # If we had no prediction, just pick the highest confidence detection
                sorted_dets = sorted(ball_detections, key=lambda x: x['confidence'], reverse=True)
                matched_det = sorted_dets[0]

        if matched_det:
            new_box = matched_det['box']
            new_center = ((new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0)
            
            # Update velocity
            if self.last_position:
                vx = new_center[0] - self.last_position[0]
                vy = new_center[1] - self.last_position[1]
                # Smooth velocity
                if self.state == 'tracked':
                    self.velocity = (
                        0.7 * vx + 0.3 * self.velocity[0],
                        0.7 * vy + 0.3 * self.velocity[1]
                    )
                else:
                    self.velocity = (vx, vy)
            else:
                self.velocity = (0.0, 0.0)

            self.last_position = new_center
            self.last_box = new_box
            self.lost_count = 0
            self.state = 'tracked'
            self.confidence = matched_det['confidence']
        else:
            # No detection matched, handle lost state
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.state = 'lost'
                self.last_position = None
                self.last_box = None
                self.velocity = (0.0, 0.0)
                self.confidence = 0.0
                return None
            else:
                # Extrapolate ball coordinates using velocity
                if self.last_position and self.last_box:
                    self.last_position = (
                        self.last_position[0] + self.velocity[0],
                        self.last_position[1] + self.velocity[1]
                    )
                    self.last_box = [
                        self.last_box[0] + self.velocity[0],
                        self.last_box[1] + self.velocity[1],
                        self.last_box[2] + self.velocity[0],
                        self.last_box[3] + self.velocity[1]
                    ]
                    # Decay velocity and confidence slowly
                    self.velocity = (self.velocity[0] * 0.9, self.velocity[1] * 0.9)
                    self.confidence *= 0.9

        return {
            'box': self.last_box,
            'center': self.last_position,
            'confidence': self.confidence,
            'state': self.state
        }
