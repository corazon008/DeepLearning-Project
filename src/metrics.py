import sklearn.metrics as skm
import numpy as np


def compute_metrics_reg(preds:np.ndarray, targets:np.ndarray):
    """
    Compute regression metrics: MSE, MAE, R2.

    Args:
        preds (np.ndarray): Predicted values.
        targets (np.ndarray): True values.
    Returns:
        dict: Dictionary containing MSE, MAE, and R2 scores.
    """
    mse = skm.mean_squared_error(preds, targets)
    mae = skm.mean_absolute_error(preds, targets)
    r2 = skm.r2_score(targets, preds)

    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

def compute_metrics_clf(preds:np.ndarray, targets:np.ndarray):
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1-score.
    Args:
        preds (np.ndarray): Predicted class probabilities or logits.
        targets (np.ndarray): True class labels.
    Returns:
        dict: Dictionary containing Accuracy, Precision, Recall, and F1-score.
    """
    #preds = np.argmax(preds, axis=1)
    accuracy = skm.accuracy_score(targets, preds)
    precision = skm.precision_score(targets, preds, average='weighted', zero_division=0)
    recall = skm.recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = skm.f1_score(targets, preds, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'F1-score': f1
    }