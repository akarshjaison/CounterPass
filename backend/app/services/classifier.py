import cv2
import numpy as np
import math
import random

def extract_jersey_color(frame, box):
    """
    Extracts the average jersey color from the player's bounding box chest area,
    excluding green background grass pixels.
    frame: numpy array in BGR format.
    box: list of floats [x_min, y_min, x_max, y_max] mapping to frame dimensions.
    Returns: (r, g, b) tuple of floats representing average jersey color.
    """
    h, w, _ = frame.shape
    x_min, y_min, x_max, y_max = box
    
    # Calculate crop coordinates focusing on upper torso (chest)
    box_h = y_max - y_min
    box_w = x_max - x_min
    
    ymin_chest = int(y_min + 0.20 * box_h)
    ymax_chest = int(y_min + 0.45 * box_h)
    xmin_chest = int(x_min + 0.35 * box_w)
    xmax_chest = int(x_min + 0.65 * box_w)
    
    # Clamp boundaries to image dimensions
    ymin_chest = max(0, min(ymin_chest, h - 1))
    ymax_chest = max(ymin_chest + 1, min(ymax_chest, h))
    xmin_chest = max(0, min(xmin_chest, w - 1))
    xmax_chest = max(xmin_chest + 1, min(xmax_chest, w))
    
    # Crop chest region
    chest = frame[ymin_chest:ymax_chest, xmin_chest:xmax_chest]
    
    if chest.size == 0:
        return 128.0, 128.0, 128.0  # Default neutral gray fallback
        
    # Convert crop to HSV to isolate green grass pixels
    hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
    
    # Define green grass HSV range bounds
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    
    # Mask out green pixels
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    
    # Select non-green pixels
    non_green_pixels = chest[non_green_mask > 0]
    
    if non_green_pixels.size == 0:
        # Fallback to general average if chest patch has only green elements
        avg_bgr = cv2.mean(chest)[:3]
    else:
        avg_bgr = np.mean(non_green_pixels, axis=0)
        
    # Convert BGR to RGB format
    return float(avg_bgr[2]), float(avg_bgr[1]), float(avg_bgr[0])

def kmeans_classify(track_colors, k=2, max_iters=50):
    """
    Clusters average colors of player tracks using K-Means (K=2) to classify teams.
    track_colors: dict mapping track_id -> [R, G, B]
    Returns: dict mapping track_id -> "Team A" | "Team B" | "Referee"
    """
    if not track_colors:
        return {}
        
    track_ids = list(track_colors.keys())
    data = [track_colors[tid] for tid in track_ids]
    
    # If we have fewer track IDs than the requested teams, default all to Team A
    if len(data) < k:
        return {tid: "Team A" for tid in track_ids}
        
    # Initialize centroid locations deterministically for repeatable clustering
    random.seed(42)
    centroids = random.sample(data, k)
    
    assignments = [0] * len(data)
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        new_assignments = []
        for x in data:
            dists = [math.sqrt(sum((x[i] - c[i])**2 for i in range(3))) for c in centroids]
            min_idx = dists.index(min(dists))
            clusters[min_idx].append(x)
            new_assignments.append(min_idx)
            
        # Compute new cluster centers
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                avg = [sum(x[j] for x in clusters[i]) / len(clusters[i]) for j in range(3)]
                new_centroids.append(avg)
            else:
                new_centroids.append(centroids[i])
                
        # Break early if centroids stabilize
        diff = sum(math.sqrt(sum((new_centroids[i][j] - centroids[i][j])**2 for j in range(3))) for i in range(k))
        assignments = new_assignments
        if diff < 1e-4:
            break
        centroids = new_centroids

    # Compute outlier metrics for Referee determination
    cluster_dists = []
    for idx, (x, assign) in enumerate(zip(data, assignments)):
        c = centroids[assign]
        dist = math.sqrt(sum((x[i] - c[i])**2 for i in range(3)))
        cluster_dists.append(dist)
        
    mean_dist = sum(cluster_dists) / len(cluster_dists) if cluster_dists else 0.0
    
    results = {}
    for idx, tid in enumerate(track_ids):
        x = track_colors[tid]
        assign = assignments[idx]
        c = centroids[assign]
        dist = math.sqrt(sum((x[i] - c[i])**2 for i in range(3)))
        
        # Outlier classification: If the track is far from its closest cluster center,
        # label as Referee (dist > 2.5 * mean_dist and dist > 60 in RGB distance space)
        if dist > 2.5 * mean_dist and dist > 60.0:
            results[tid] = "Referee"
        else:
            results[tid] = "Team A" if assign == 0 else "Team B"
            
    return results
