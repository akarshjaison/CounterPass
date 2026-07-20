import pytest
from app.services.tracker import BallTracker
from app.services.buffer import TemporalFrameBuffer

def test_ball_tracker_initial_tracking():
    tracker = BallTracker(max_lost_frames=3)
    assert tracker.state == 'lost'
    
    # Frame 1: Initial ball detection
    state = tracker.update([{'box': [100.0, 100.0, 110.0, 110.0], 'confidence': 0.85}])
    assert state is not None
    assert tracker.state == 'tracked'
    assert tracker.last_position == (105.0, 105.0)
    assert tracker.velocity == (0.0, 0.0)
    
    # Frame 2: Ball moves (e.g., dx = 10, dy = 20)
    # Since state was 'tracked', velocity is smoothed: 0.7 * 10 + 0.3 * 0 = 7.0, and 0.7 * 20 = 14.0
    state = tracker.update([{'box': [110.0, 120.0, 120.0, 130.0], 'confidence': 0.9}])
    assert state is not None
    assert tracker.state == 'tracked'
    assert tracker.last_position == (115.0, 125.0)
    assert tracker.velocity == (7.0, 14.0)

def test_ball_tracker_occlusion_prediction():
    tracker = BallTracker(max_lost_frames=2)
    
    # Frame 1 & 2: Set up a velocity vector
    tracker.update([{'box': [10.0, 10.0, 20.0, 20.0], 'confidence': 0.9}])
    tracker.update([{'box': [20.0, 30.0, 30.0, 40.0], 'confidence': 0.9}])
    
    # Velocity is smoothed: (7.0, 14.0)
    assert tracker.velocity == (7.0, 14.0)
    
    # Frame 3: Ball is occluded (empty detection list)
    # Tracker should predict: (25.0 + 7.0, 35.0 + 14.0) = (32.0, 49.0)
    state = tracker.update([])
    assert state is not None
    assert state['state'] == 'tracked'
    assert tracker.last_position == (32.0, 49.0)
    assert tracker.lost_count == 1
    
    # Frame 4: Ball still occluded
    # velocity decays: 7 * 0.9 = 6.3, 14 * 0.9 = 12.6
    # next prediction: (32.0 + 6.3, 49.0 + 12.6) = (38.3, 61.6)
    state = tracker.update([])
    assert state is not None
    assert tracker.lost_count == 2
    assert tracker.last_position == pytest.approx((38.3, 61.6))
    
    # Frame 5: Occluded count (3) exceeds max_lost_frames (2). Returns None (lost).
    state = tracker.update([])
    assert state is None
    assert tracker.state == 'lost'

def test_buffer_ball_interpolation_and_smoothing():
    buffer = TemporalFrameBuffer(size=10)
    
    # Add Frame 0 with ball at (10, 10)
    buffer.add_frame(
        frame_index=0, timestamp=0.0,
        player_detections=[],
        ball_detection={'box': [5, 5, 15, 15], 'center': (10.0, 10.0), 'confidence': 0.9}
    )
    # Add Frame 2 with ball at (30, 30) (Frame 1 is missing/occluded)
    buffer.add_frame(
        frame_index=2, timestamp=0.2,
        player_detections=[],
        ball_detection={'box': [25, 25, 35, 35], 'center': (30.0, 30.0), 'confidence': 0.8}
    )
    
    # Get raw trajectory
    raw = buffer.get_ball_trajectory()
    assert len(raw) == 2
    
    # Interpolate gaps
    interpolated = buffer.interpolate_ball_gaps()
    assert len(interpolated) == 3
    assert interpolated[1]['frame_index'] == 1
    assert interpolated[1]['center'] == pytest.approx((20.0, 20.0))
    assert interpolated[1]['source'] == 'interpolated'
    
    # Add frame 3 and 4 to test smoothing (requires length >= window_size)
    buffer.add_frame(
        frame_index=3, timestamp=0.3,
        player_detections=[],
        ball_detection={'box': [35, 35, 45, 45], 'center': (40.0, 40.0), 'confidence': 0.8}
    )
    buffer.add_frame(
        frame_index=4, timestamp=0.4,
        player_detections=[],
        ball_detection={'box': [45, 45, 55, 55], 'center': (50.0, 50.0), 'confidence': 0.8}
    )
    
    # Add frame 5 (total 5 frames)
    buffer.add_frame(
        frame_index=5, timestamp=0.5,
        player_detections=[],
        ball_detection={'box': [55, 55, 65, 65], 'center': (60.0, 60.0), 'confidence': 0.8}
    )
    
    smoothed = buffer.smooth_ball_trajectory(window_size=3)
    assert len(smoothed) == 6 # frames 0, 1, 2, 3, 4, 5
    # Smooth center at index 3 (average of index 2, 3, 4: (30, 40, 50) / 3 = 40)
    assert smoothed[3]['center'] == pytest.approx((40.0, 40.0))
