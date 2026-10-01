"""
Configuration file for the Wine Classification pipeline.
Centralizes random seeds, paths, and model hyperparameters.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Random Seed for reproducibility
RANDOM_STATE = 42

# Training setup
TEST_SIZE = 0.2

# Target accuracy threshold for the decision loop
TARGET_ACCURACY = 0.90
