import pytest
from app.services.buffer import TemporalFrameBuffer

def test_buffer_add_and_sliding_window():
    """
    Test that TemporalFrameBuffer stores frames and enforces the sliding window size.
    """
    buffer = TemporalFrameBuffer(size=3)
    
    # Add 4 frames (exceeding max size of 3)
    for i in range(4):
        buffer.add_frame(
            frame_index=i,
            timestamp=i * 0.1,
            player_detections=[{'track_id': 1, 'box': [10.0, 10.0, 20.0, 20.0], 'center': (15.0, 15.0), 'confidence': 0.9}]
        )
        
    assert len(buffer.buffer) == 3
    # The oldest frame (index 0) should be evicted
    assert buffer.buffer[0]['frame_index'] == 1
    assert buffer.buffer[-1]['frame_index'] == 3

def test_buffer_trajectory_retrieval():
    """
    Test that we can retrieve the trajectory of a specific track from the buffer.
    """
    buffer = TemporalFrameBuffer(size=10)
    buffer.add_frame(
        frame_index=0, timestamp=0.0,
        player_detections=[
            {'track_id': 1, 'box': [10, 10, 20, 20], 'center': (15, 15), 'confidence': 0.9},
            {'track_id': 2, 'box': [30, 30, 40, 40], 'center': (35, 35), 'confidence': 0.8}
        ]
    )
    buffer.add_frame(
        frame_index=1, timestamp=0.1,
        player_detections=[
            {'track_id': 1, 'box': [12, 12, 22, 22], 'center': (17, 17), 'confidence': 0.91}
        ]
    )
    
    traj1 = buffer.get_trajectory(1)
    traj2 = buffer.get_trajectory(2)
    
    assert len(traj1) == 2
    assert traj1[0]['center'] == (15, 15)
    assert traj1[1]['center'] == (17, 17)
    
    assert len(traj2) == 1
    assert traj2[0]['center'] == (35, 35)

def test_buffer_linear_interpolation():
    """
    Test that coordinate gaps are filled correctly using linear interpolation.
    """
    buffer = TemporalFrameBuffer(size=10)
    
    # Frame 0: Player 1 present
    buffer.add_frame(
        frame_index=0, timestamp=0.0,
        player_detections=[{'track_id': 1, 'box': [10.0, 10.0, 20.0, 20.0], 'center': (15.0, 15.0), 'confidence': 0.8}]
    )
    # Frame 3: Player 1 present again (gap of 2 frames: 1, 2)
    buffer.add_frame(
        frame_index=3, timestamp=0.3,
        player_detections=[{'track_id': 1, 'box': [40.0, 40.0, 50.0, 50.0], 'center': (45.0, 45.0), 'confidence': 0.9}]
    )
    
    traj = buffer.interpolate_missing_frames(track_id=1, max_gap=5)
    
    # Total frames should be 4 (indexes 0, 1, 2, 3)
    assert len(traj) == 4
    assert traj[0]['frame_index'] == 0
    assert traj[1]['frame_index'] == 1
    assert traj[2]['frame_index'] == 2
    assert traj[3]['frame_index'] == 3
    
    # Check linear interpolation of coordinates:
    # center_x at index 0 is 15.0, index 3 is 45.0
    # Step = 10.0 per frame (15.0 -> 25.0 -> 35.0 -> 45.0)
    assert traj[1]['center'] == pytest.approx((25.0, 25.0))
    assert traj[2]['center'] == pytest.approx((35.0, 35.0))
    
    # Check box coordinates at index 2
    # xmin: 10.0 -> 20.0 -> 30.0 -> 40.0
    assert traj[2]['box'] == pytest.approx([30.0, 30.0, 40.0, 40.0])

def test_buffer_trajectory_smoothing():
    """
    Test moving average smoothing on noisy trajectory coordinates.
    """
    buffer = TemporalFrameBuffer(size=10)
    # Generate 5 consecutive frames with a single coordinate spike (noise)
    for i, cy in enumerate([10.0, 11.0, 100.0, 13.0, 14.0]):
        buffer.add_frame(
            frame_index=i, timestamp=i * 0.1,
            player_detections=[{'track_id': 1, 'box': [10, cy-5, 20, cy+5], 'center': (15.0, cy), 'confidence': 0.85}]
        )
        
    smoothed = buffer.smooth_trajectory(track_id=1, window_size=3)
    
    assert len(smoothed) == 5
    # The coordinate spike (100.0) at index 2 should be smoothed by averaging index 1, 2, 3:
    # (11.0 + 100.0 + 13.0) / 3 = 124.0 / 3 = 41.333
    assert smoothed[2]['center'][1] == pytest.approx(41.3333, rel=1e-4)

def test_buffer_velocity_estimation():
    """
    Test average velocity calculation.
    """
    buffer = TemporalFrameBuffer(size=10)
    buffer.add_frame(
        frame_index=1, timestamp=0.1,
        player_detections=[{'track_id': 1, 'box': [10, 10, 20, 20], 'center': (15.0, 15.0), 'confidence': 0.9}]
    )
    buffer.add_frame(
        frame_index=2, timestamp=0.2,
        player_detections=[{'track_id': 1, 'box': [12, 13, 22, 23], 'center': (17.0, 18.0), 'confidence': 0.9}]
    )
    buffer.add_frame(
        frame_index=3, timestamp=0.3,
        player_detections=[{'track_id': 1, 'box': [14, 16, 24, 26], 'center': (19.0, 21.0), 'confidence': 0.9}]
    )
    
    vx, vy = buffer.get_average_velocity(track_id=1, num_frames=3)
    
    # Velocity dx/df = (19.0 - 15.0) / (3 - 1) = 4.0 / 2 = 2.0
    # dy/df = (21.0 - 15.0) / 2 = 6.0 / 2 = 3.0
    assert vx == pytest.approx(2.0)
    assert vy == pytest.approx(3.0)
