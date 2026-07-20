from collections import deque

class TemporalFrameBuffer:
    def __init__(self, size: int = 60):
        """
        Initializes the sliding window temporal buffer.
        size: The maximum number of historical frames to retain.
        """
        self.size = size
        self.buffer = deque(maxlen=size)

    def add_frame(self, frame_index: int, timestamp: float, player_detections: list, ball_detection: dict = None):
        """
        Adds a new frame's tracking data to the buffer.
        player_detections: list of dicts: {
            'track_id': int,
            'box': [xmin, ymin, xmax, ymax],
            'center': (cx, cy),
            'confidence': float
        }
        ball_detection: dict: {
            'box': [xmin, ymin, xmax, ymax],
            'center': (cx, cy),
            'confidence': float
        } (or None if ball not detected in the frame)
        """
        # Convert list of player detections to a mapping for fast ID-based lookups
        players_map = {}
        for det in player_detections:
            track_id = det.get('track_id')
            if track_id is not None:
                players_map[track_id] = {
                    'box': list(det['box']),
                    'center': tuple(det['center']),
                    'confidence': float(det['confidence'])
                }

        frame_data = {
            'frame_index': frame_index,
            'timestamp': timestamp,
            'players': players_map,
            'ball': ball_detection if ball_detection is None else {
                'box': list(ball_detection['box']),
                'center': tuple(ball_detection['center']),
                'confidence': float(ball_detection['confidence'])
            }
        }
        self.buffer.append(frame_data)

    def get_trajectory(self, track_id: int) -> list:
        """
        Retrieves the observed history of a specific player track present in the buffer.
        Returns: list of dicts containing frame tracking details.
        """
        trajectory = []
        for frame in self.buffer:
            if track_id in frame['players']:
                p_data = frame['players'][track_id]
                trajectory.append({
                    'frame_index': frame['frame_index'],
                    'timestamp': frame['timestamp'],
                    'box': p_data['box'],
                    'center': p_data['center'],
                    'confidence': p_data['confidence']
                })
        return trajectory

    def interpolate_missing_frames(self, track_id: int, max_gap: int = 15) -> list:
        """
        Checks for gaps in a player's trajectory and fills them using linear interpolation.
        max_gap: The maximum number of consecutive missing frames allowed for interpolation.
        Returns: list of dicts including both observed and interpolated states.
        """
        observed = self.get_trajectory(track_id)
        if len(observed) < 2:
            return observed

        interpolated = []
        for i in range(len(observed) - 1):
            curr_pt = observed[i]
            next_pt = observed[i + 1]
            interpolated.append(curr_pt)

            gap = next_pt['frame_index'] - curr_pt['frame_index']
            if 1 < gap <= max_gap:
                # Perform linear interpolation across the gap
                for step in range(1, gap):
                    t = step / float(gap)
                    
                    # Interpolated frame index and timestamp
                    f_idx = curr_pt['frame_index'] + step
                    timestamp = curr_pt['timestamp'] + t * (next_pt['timestamp'] - curr_pt['timestamp'])
                    
                    # Interpolate center (x, y)
                    cx = curr_pt['center'][0] + t * (next_pt['center'][0] - curr_pt['center'][0])
                    cy = curr_pt['center'][1] + t * (next_pt['center'][1] - curr_pt['center'][1])
                    
                    # Interpolate bounding box [xmin, ymin, xmax, ymax]
                    box = [
                        curr_pt['box'][j] + t * (next_pt['box'][j] - curr_pt['box'][j])
                        for j in range(4)
                    ]
                    
                    # Interpolated confidence decays slightly based on the step from observed
                    confidence = min(curr_pt['confidence'], next_pt['confidence']) * 0.95

                    interpolated.append({
                        'frame_index': f_idx,
                        'timestamp': timestamp,
                        'box': box,
                        'center': (cx, cy),
                        'confidence': confidence,
                        'source': 'interpolated'
                    })
        
        # Add the last observed point
        if observed:
            interpolated.append(observed[-1])
            
        return interpolated

    def smooth_trajectory(self, track_id: int, window_size: int = 5) -> list:
        """
        Applies a moving average smoothing filter to a player's trajectory.
        window_size: Odd integer representing the size of the smoothing window.
        Returns: list of dicts with smoothed center and box coordinates.
        """
        # Always interpolate gaps first to ensure a continuous stream of frames
        trajectory = self.interpolate_missing_frames(track_id)
        if len(trajectory) < window_size:
            return trajectory

        half_w = window_size // 2
        smoothed = []

        for i in range(len(trajectory)):
            # Boundaries clamping for window
            start_idx = max(0, i - half_w)
            end_idx = min(len(trajectory), i + half_w + 1)
            window_pts = trajectory[start_idx:end_idx]

            # Compute average center
            sum_x = sum(pt['center'][0] for pt in window_pts)
            sum_y = sum(pt['center'][1] for pt in window_pts)
            avg_center = (sum_x / len(window_pts), sum_y / len(window_pts))

            # Compute average box
            avg_box = [0.0, 0.0, 0.0, 0.0]
            for j in range(4):
                avg_box[j] = sum(pt['box'][j] for pt in window_pts) / len(window_pts)

            # Copy details and insert smoothed coordinates
            pt_copy = dict(trajectory[i])
            pt_copy['center'] = avg_center
            pt_copy['box'] = avg_box
            smoothed.append(pt_copy)

        return smoothed

    def get_average_velocity(self, track_id: int, num_frames: int = 5) -> tuple:
        """
        Calculates the average velocity vector (vx, vy) in pixels/frame over the last num_frames.
        Returns: (vx, vy) tuple.
        """
        trajectory = self.get_trajectory(track_id)
        if len(trajectory) < 2:
            return 0.0, 0.0

        # Grab the last available coordinates up to num_frames
        pts = trajectory[-min(len(trajectory), num_frames):]
        
        total_dx = 0.0
        total_dy = 0.0
        total_df = 0
        
        for i in range(len(pts) - 1):
            curr_pt = pts[i]
            next_pt = pts[i + 1]
            df = next_pt['frame_index'] - curr_pt['frame_index']
            if df > 0:
                dx = next_pt['center'][0] - curr_pt['center'][0]
                dy = next_pt['center'][1] - curr_pt['center'][1]
                total_dx += dx
                total_dy += dy
                total_df += df

        if total_df == 0:
            return 0.0, 0.0

        return total_dx / total_df, total_dy / total_df

    def get_ball_trajectory(self) -> list:
        """
        Retrieves the observed history of the ball present in the buffer.
        """
        trajectory = []
        for frame in self.buffer:
            if frame.get('ball') is not None:
                b_data = frame['ball']
                trajectory.append({
                    'frame_index': frame['frame_index'],
                    'timestamp': frame['timestamp'],
                    'box': b_data['box'],
                    'center': b_data['center'],
                    'confidence': b_data['confidence']
                })
        return trajectory

    def interpolate_ball_gaps(self, max_gap: int = 15) -> list:
        """
        Fills in missing ball positions across frames using linear interpolation.
        """
        observed = self.get_ball_trajectory()
        if len(observed) < 2:
            return observed

        interpolated = []
        for i in range(len(observed) - 1):
            curr_pt = observed[i]
            next_pt = observed[i + 1]
            interpolated.append(curr_pt)

            gap = next_pt['frame_index'] - curr_pt['frame_index']
            if 1 < gap <= max_gap:
                for step in range(1, gap):
                    t = step / float(gap)
                    f_idx = curr_pt['frame_index'] + step
                    timestamp = curr_pt['timestamp'] + t * (next_pt['timestamp'] - curr_pt['timestamp'])
                    
                    cx = curr_pt['center'][0] + t * (next_pt['center'][0] - curr_pt['center'][0])
                    cy = curr_pt['center'][1] + t * (next_pt['center'][1] - curr_pt['center'][1])
                    
                    box = [
                        curr_pt['box'][j] + t * (next_pt['box'][j] - curr_pt['box'][j])
                        for j in range(4)
                    ]
                    confidence = min(curr_pt['confidence'], next_pt['confidence']) * 0.95

                    interpolated.append({
                        'frame_index': f_idx,
                        'timestamp': timestamp,
                        'box': box,
                        'center': (cx, cy),
                        'confidence': confidence,
                        'source': 'interpolated'
                    })
        
        if observed:
            interpolated.append(observed[-1])
        return interpolated

    def smooth_ball_trajectory(self, window_size: int = 5) -> list:
        """
        Applies a moving average smoothing filter to the ball's trajectory.
        """
        trajectory = self.interpolate_ball_gaps()
        if len(trajectory) < window_size:
            return trajectory

        half_w = window_size // 2
        smoothed = []

        for i in range(len(trajectory)):
            start_idx = max(0, i - half_w)
            end_idx = min(len(trajectory), i + half_w + 1)
            window_pts = trajectory[start_idx:end_idx]

            sum_x = sum(pt['center'][0] for pt in window_pts)
            sum_y = sum(pt['center'][1] for pt in window_pts)
            avg_center = (sum_x / len(window_pts), sum_y / len(window_pts))

            avg_box = [0.0, 0.0, 0.0, 0.0]
            for j in range(4):
                avg_box[j] = sum(pt['box'][j] for pt in window_pts) / len(window_pts)

            pt_copy = dict(trajectory[i])
            pt_copy['center'] = avg_center
            pt_copy['box'] = avg_box
            smoothed.append(pt_copy)

        return smoothed
