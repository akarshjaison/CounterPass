import pytest
from eval.eval_harness import evaluate_pass_events, evaluate_position_accuracy, evaluate_team_classification

def test_eval_pass_events():
    pred = [{"timestamp": 5.2, "passer_track_id": 1, "receiver_track_id": 2}]
    gt = [{"timestamp": 5.0, "passer_track_id": 1, "receiver_track_id": 2}]
    metrics = evaluate_pass_events(pred, gt, time_tolerance_sec=1.0)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0

def test_eval_position_accuracy():
    pred = [(10.0, 20.0)]
    gt = [(10.0, 20.0)]
    metrics = evaluate_position_accuracy(pred, gt)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0

def test_eval_team_classification():
    pred = {1: "Team A", 2: "Team B"}
    gt = {1: "Team A", 2: "Team B"}
    metrics = evaluate_team_classification(pred, gt)

    assert metrics["accuracy"] == 1.0
