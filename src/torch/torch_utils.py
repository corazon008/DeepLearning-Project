import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

device = "cpu"

def compute_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    return torch.tensor(weights, dtype=torch.float32)
