import os
import sys

# Setup mock environment
sys.path.insert(0, os.path.abspath("backend"))
from app.services.calibration import PitchCalibrator
from app.services.analytics_engine import run_tactical_analysis
from app.services.annotator import annotate_video

print("Imports successful!")

# Test sports availability
try:
    from sports.configs.soccer import SoccerPitchConfiguration
    print("Sports Config loaded!")
except ImportError as e:
    print(f"Sports import error: {e}")

# Quick initialization of calibrator
calibrator = PitchCalibrator()
print(f"Pitch Calibrator initialized: {calibrator}")

print("Verification check complete.")
