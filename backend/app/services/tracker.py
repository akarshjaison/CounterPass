import math
import numpy as np
import supervision as sv

def compute_iou(box1, box2):
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
    def __init__(self, max_lost_frames=30, max_tracks=30):
        # We replace custom tracking with supervision.ByteTrack
        self.byte_tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=max_lost_frames, minimum_matching_threshold=0.8, frame_rate=30)
        self.tracks = {}  # track_id -> track metadata

    def update(self, detections):
        if not detections:
            sv_dets = sv.Detections.empty()
        else:
            xyxy = np.array([d['box'] for d in detections])
            confidence = np.array([d['confidence'] for d in detections])
            class_id = np.zeros(len(detections), dtype=int)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            
        tracked_dets = self.byte_tracker.update_with_detections(sv_dets)
        
        active_tids = set()
        active_tracks = []
        
        tracker_ids = tracked_dets.tracker_id
        if tracker_ids is None:
            tracker_ids = np.zeros(len(tracked_dets), dtype=int)
            
        for i in range(len(tracked_dets)):
            box = tracked_dets.xyxy[i].tolist()
            tid = int(tracker_ids[i])
            active_tids.add(tid)
            # tracked_dets confidence can sometimes be missing or none in sv, fall back to max
            conf = float(tracked_dets.confidence[i]) if tracked_dets.confidence is not None and len(tracked_dets.confidence) > i else 0.8
            
            # Map back to original detection to grab color and cls_name
            color = None
            cls_name = 'player'
            if detections:
                best_iou = 0
                for d in detections:
                    iou = compute_iou(box, d['box'])
                    if iou > best_iou:
                        best_iou = iou
                        color = d.get('color')
                        cls_name = d.get('cls_name', 'player')
            
            if color is not None:
                if tid in self.tracks and self.tracks[tid].get('avg_color') is not None:
                    prev_color = self.tracks[tid]['avg_color']
                    avg_color = tuple(0.85 * prev_color[j] + 0.15 * color[j] for j in range(3))
                else:
                    avg_color = color
            else:
                avg_color = self.tracks.get(tid, {}).get('avg_color')
                cls_name = self.tracks.get(tid, {}).get('cls_name', 'player')

            track_info = {
                'id': tid,
                'box': box,
                'center': ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                'confidence': conf,
                'avg_color': avg_color,
                'cls_name': cls_name,
                'state': 'tracked'
            }
            self.tracks[tid] = track_info
            active_tracks.append(track_info)
            
        for tid in self.tracks:
            if tid not in active_tids:
                self.tracks[tid]['state'] = 'lost'
                
        return active_tracks


class BallTracker:
    def __init__(self, max_lost_frames=10):
        self.max_lost_frames = max_lost_frames
        self.last_position = None
        self.last_box = None
        self.velocity = (0.0, 0.0)
        self.lost_count = 0
        self.state = 'lost'
        self.confidence = 0.0

    def update(self, ball_detections):
        if self.state == 'lost' and not ball_detections:
            return None

        pred_pos = None
        if self.state == 'tracked' and self.last_position:
            pred_pos = (
                self.last_position[0] + self.velocity[0],
                self.last_position[1] + self.velocity[1]
            )

        matched_det = None
        if ball_detections:
            if pred_pos:
                dists = []
                for det in ball_detections:
                    box = det['box']
                    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                    dist = math.sqrt((center[0] - pred_pos[0])**2 + (center[1] - pred_pos[1])**2)
                    dists.append((dist, det))
                dists.sort(key=lambda x: x[0])
                if dists[0][0] < 150.0:
                    matched_det = dists[0][1]
            else:
                sorted_dets = sorted(ball_detections, key=lambda x: x['confidence'], reverse=True)
                matched_det = sorted_dets[0]

        if matched_det:
            new_box = matched_det['box']
            new_center = ((new_box[0] + new_box[2]) / 2.0, (new_box[1] + new_box[3]) / 2.0)
            
            if self.last_position:
                vx = new_center[0] - self.last_position[0]
                vy = new_center[1] - self.last_position[1]
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
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.state = 'lost'
                self.last_position = None
                self.last_box = None
                self.velocity = (0.0, 0.0)
                self.confidence = 0.0
                return None
            else:
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
                    self.velocity = (self.velocity[0] * 0.9, self.velocity[1] * 0.9)
                    self.confidence *= 0.9

        return {
            'box': self.last_box,
            'center': self.last_position,
            'confidence': self.confidence,
            'state': self.state
        }
