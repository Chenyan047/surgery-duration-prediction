"""
Configuration file for Surgery Duration Prediction project.
Contains all project settings, paths, and parameters.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Source code directory
SRC_DIR = PROJECT_ROOT / "src"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "surgery_hernia_data_set.xlsx"

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Deep learning parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Feature engineering
NUMERICAL_FEATURES = [
    # Add your numerical feature names here
]

CATEGORICAL_FEATURES = [
    # Add your categorical feature names here
]

TARGET_FEATURE = "surgery_duration"  # Adjust based on your dataset

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Call this function when the module is imported
ensure_directories()
