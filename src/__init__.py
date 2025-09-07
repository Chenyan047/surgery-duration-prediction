"""
Surgery Duration Prediction Package

This package provides tools for predicting surgery duration using both
traditional machine learning and deep learning approaches.
"""

import os
import numpy as np
import torch

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    """
    Set random seeds for all random number generators to ensure reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set PyTorch deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Automatically set random seeds when package is imported
set_random_seeds()

__version__ = "0.1.0"
__author__ = "Surgery Duration Prediction Team"
