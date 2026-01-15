
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time

def get_hammersley_sequence(n_samples, scramble=True):
    """Generates 2D Hammersley points in [0,1]^2"""
    n = np.arange(n_samples)
    y = np.zeros(n_samples)
    seed = n + 1 # 1-based index
    base_inv = 0.5
    while np.any(seed > 0):
        y += (seed % 2) * base_inv
        seed //= 2
        base_inv /= 2

    x = (n + 0.5) / n_samples
    
    points = np.stack([x, y], axis=1) # Shape (N, 2)

    if scramble:
        shift = np.random.rand(2)
        points = (points + shift) % 1.0
        
    return torch.tensor(points, dtype=torch.float32)

