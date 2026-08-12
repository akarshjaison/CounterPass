import os
import json
import math
import sys
from typing import List, Dict, Any, Tuple

def evaluate_pass_events(
    predicted_passes: List[Dict[str, Any]],
    ground_truth_passes: List[Dict[str, Any]],
    time_tolerance_sec: float = 1.0
) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1-score for detected pass events against ground-truth passes.
    """
    if not ground_truth_passes:
        return {"precision": 1.0 if not predicted_passes else 0.0, "recall": 1.0, "f1_score": 1.0}

    true_positives = 0
    matched_gt = set()

    for pred in predicted_passes:
        pred_ts = pred.get("timestamp", 0.0)
        pred_passer = pred.get("passer_track_id")
        pred_receiver = pred.get("receiver_track_id")

        for idx, gt in enumerate(ground_truth_passes):
            if idx in matched_gt:
                continue
            gt_ts = gt.get("timestamp", 0.0)
            gt_passer = gt.get("passer_track_id")
            gt_receiver = gt.get("receiver_track_id")

            if abs(pred_ts - gt_ts) <= time_tolerance_sec:
                if pred_passer == gt_passer and pred_receiver == gt_receiver:
                    true_positives += 1
                    matched_gt.add(idx)
                    break

    precision = true_positives / len(predicted_passes) if predicted_passes else 0.0
    recall = true_positives / len(ground_truth_passes) if ground_truth_passes else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": true_positives,
        "total_predicted": len(predicted_passes),
        "total_ground_truth": len(ground_truth_passes)
    }

def evaluate_position_accuracy(
    predicted_positions: List[Tuple[float, float]],
    ground_truth_positions: List[Tuple[float, float]]
) -> Dict[str, float]:
    """
    Calculates Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) in position coordinates.
    """
    if not predicted_positions or not ground_truth_positions:
        return {"mae": 0.0, "rmse": 0.0}

    n = min(len(predicted_positions), len(ground_truth_positions))
    errors = []
    sq_errors = []

    for i in range(n):
        px, py = predicted_positions[i]
        gx, gy = ground_truth_positions[i]
        err = math.sqrt((px - gx)**2 + (py - gy)**2)
        errors.append(err)
        sq_errors.append(err**2)

    mae = sum(errors) / n if n > 0 else 0.0
    rmse = math.sqrt(sum(sq_errors) / n) if n > 0 else 0.0

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "samples_evaluated": n}

def evaluate_team_classification(
    predicted_teams: Dict[int, str],
    ground_truth_teams: Dict[int, str]
) -> Dict[str, float]:
    """
    Calculates team classification accuracy percentage.
    """
    if not ground_truth_teams:
        return {"accuracy": 1.0}

    correct = 0
    total = 0
    for tid, gt_team in ground_truth_teams.items():
        if tid in predicted_teams:
            total += 1
            if predicted_teams[tid] == gt_team:
                correct += 1

    acc = (correct / total) if total > 0 else 0.0
    return {"accuracy": round(acc, 4), "correct_matches": correct, "total_matched": total}

def run_evaluation(dataset_file: str) -> Dict[str, Any]:
    """
    Runs evaluation against a ground-truth JSON dataset file.
    """
    if not os.path.exists(dataset_file):
        print(f"[Eval] Dataset file not found: {dataset_file}")
        return {"status": "error", "message": f"File not found: {dataset_file}"}

    with open(dataset_file, "r") as f:
        data = json.load(f)

    pred_passes = data.get("predicted_passes", [])
    gt_passes = data.get("ground_truth_passes", [])
    pass_metrics = evaluate_pass_events(pred_passes, gt_passes)

    pred_pos = data.get("predicted_positions", [])
    gt_pos = data.get("ground_truth_positions", [])
    pos_metrics = evaluate_position_accuracy(pred_pos, gt_pos)

    pred_teams = data.get("predicted_teams", {})
    gt_teams = data.get("ground_truth_teams", {})
    team_metrics = evaluate_team_classification(pred_teams, gt_teams)

    report = {
        "pass_metrics": pass_metrics,
        "position_metrics": pos_metrics,
        "team_metrics": team_metrics
    }
    
    print("\n================ COUNTERPASS EVALUATION REPORT ================")
    print(f"Pass Detection Precision: {pass_metrics['precision']:.2%}")
    print(f"Pass Detection Recall:    {pass_metrics['recall']:.2%}")
    print(f"Pass Detection F1 Score:  {pass_metrics['f1_score']:.4f}")
    print(f"Position Error (MAE):     {pos_metrics['mae']:.2f}")
    print(f"Team Class Accuracy:     {team_metrics['accuracy']:.2%}")
    print("===============================================================\n")
    
    return report

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "backend/eval/sample_gt.json"
    run_evaluation(filepath)
