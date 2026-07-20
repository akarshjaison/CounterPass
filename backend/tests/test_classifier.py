import pytest
import numpy as np
from app.services.classifier import extract_jersey_color, kmeans_classify

def test_jersey_color_extraction():
    """
    Test that extract_jersey_color correctly crops a player torso, 
    ignores green background grass pixels, and computes the jersey color.
    """
    # Create a mock 100x100 BGR frame filled with green grass color (0, 150, 0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [0, 150, 0]
    
    # Place a solid Red jersey (0, 0, 255 BGR) in the player torso region:
    # Torso region: Y: 15% to 50% of the box, X: 20% to 80% of the box
    # If player box is [10, 10, 90, 90] (size 80x80):
    # Torso Y: 10 + 12 = 22 to 10 + 40 = 50
    # Torso X: 10 + 16 = 26 to 10 + 48 = 58
    frame[22:50, 26:58] = [0, 0, 255] # Red in BGR
    
    box = [10.0, 10.0, 90.0, 90.0]
    
    r, g, b = extract_jersey_color(frame, box)
    
    # Assert that the extracted color is close to Pure Red [255, 0, 0]
    # and has filtered out the green grass background [0, 150, 0]
    assert r > 240
    assert g < 15
    assert b < 15

def test_kmeans_clustering():
    """
    Test that kmeans_classify correctly segments Red and Blue jersey tracks into separate teams.
    """
    track_colors = {
        # Team 1: Red players (with slight variations)
        1: [250, 10, 15],
        2: [255, 5, 5],
        3: [240, 12, 10],
        4: [245, 8, 8],
        5: [252, 2, 2],
        
        # Team 2: Blue players (with slight variations)
        11: [5, 10, 250],
        12: [12, 5, 245],
        13: [8, 12, 240],
        14: [2, 2, 255],
        15: [10, 8, 252]
    }
    
    classifications = kmeans_classify(track_colors, k=2)
    
    # Check that Team A and Team B groupings are homogenous
    team_1_classes = [classifications[i] for i in [1, 2, 3, 4, 5]]
    team_2_classes = [classifications[i] for i in [11, 12, 13, 14, 15]]
    
    # All Team 1 players should share the same class
    assert len(set(team_1_classes)) == 1
    # All Team 2 players should share the same class
    assert len(set(team_2_classes)) == 1
    # Team 1 class should be different from Team 2 class
    assert team_1_classes[0] != team_2_classes[0]
    
    # Values should be Team A and Team B
    assert set(team_1_classes + team_2_classes) == {"Team A", "Team B"}

def test_referee_detection():
    """
    Test that outliers (like a bright Yellow referee) are correctly classified as Referee.
    """
    track_colors = {
        # Team 1: Red
        1: [255, 0, 0],
        2: [250, 5, 5],
        3: [245, 10, 10],
        
        # Team 2: Blue
        11: [0, 0, 255],
        12: [5, 5, 250],
        13: [10, 10, 245],
        
        # Outlier: Bright Yellow Referee [255, 255, 0]
        99: [255, 255, 0]
    }
    
    classifications = kmeans_classify(track_colors, k=2)
    
    # Yellow Referee is far from Red and Blue, so it should be classified as Outlier -> "Referee"
    assert classifications[99] == "Referee"
    
    # Standard players should still be classified into teams
    assert classifications[1] in ["Team A", "Team B"]
    assert classifications[11] in ["Team A", "Team B"]
    assert classifications[1] != classifications[11]
