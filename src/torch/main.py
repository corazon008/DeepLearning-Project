import sklearn.metrics
import torch
from torch.utils.data import TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
# Move one step in the directory structure to access src
import sys
sys.path.append(os.path.abspath(os.path.join('..').join("..")))
from src.torch.torch_utils import *
from src.torch.torch_wrapper import *
from src.metrics import *


if __name__ == "__main__":
    df = pd.read_csv('../../data/health_lifestyle_dataset_cleaned.csv')

    regression_target = ['cholesterol', 'calories_consumed']
    classification_target = 'disease_risk'

    regression_features = df.drop(columns=regression_target).values
    classification_features = df.drop(columns=classification_target).values
    regression_labels = df[regression_target].values
    classification_labels = df[classification_target].values

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        regression_features, regression_labels, test_size=0.2, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        classification_features, classification_labels, test_size=0.2, random_state=42
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("Regression : -----------------------------")
    param_grid_reg = {
        "layers": [3, 4],
        "width": [32, 64],
        "lr": [1e-3],
        "epochs": [20],
        # "loss_fn": [nn.MSELoss, nn.HuberLoss],
        # "optimizer": [torch.optim.SGD, torch.optim.Adam], #
        "activation": [nn.Tanh],
        "batch_size": [16, 32],
        #"dropout_rates": [[0.3], [0.5, 0.2]],
    }

    grid_reg = GridSearchCV(TorchRegressor(), param_grid_reg, cv=3, scoring="r2", n_jobs=-1, verbose=2)
    grid_reg.fit(X_train_reg, y_train_reg)

    print("Meilleurs paramètres :", grid_reg.best_params_)
    print("Score :", grid_reg.best_score_)


    best_model = grid_reg.best_estimator_
    hist = best_model.loss_history

    y_preds_test_reg = best_model.predict(X_test_reg)
    y_preds_train_reg = best_model.predict(X_train_reg)

    test_metrics_reg = compute_metrics_reg(y_preds_test_reg, y_test_reg)
    train_metrics_reg = compute_metrics_reg(y_preds_train_reg, y_train_reg)

    print("Train Metrics Regression :", train_metrics_reg)
    print("Test Metrics Regression :", test_metrics_reg)

    plt.figure(figsize=(10, 6))
    plt.plot(hist, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs (Regression)')
    plt.legend()
    plt.show()


    print("Classification : -----------------------------")
    param_grid_clf = {
        "layers": [4, 5],
        "width": [128, 256],
        "lr": [1e-3],
        "epochs": [30],
        "class_weights": [torch.tensor([1.0, 4.5])],
        "loss_fn": [nn.CrossEntropyLoss],
        "optimizer": [torch.optim.SGD],
        "activation": [nn.ReLU],
        "batch_size": [16],
        #"dropout_rates": [[0.3], [0.5, 0.2]],
    }

    grid_clf = GridSearchCV(TorchClassifier(), param_grid_clf, cv=skf, scoring="accuracy", n_jobs=-1, verbose=2)
    grid_clf.fit(X_train_clf, y_train_clf)

    print("Meilleurs paramètres :", grid_clf.best_params_)
    print("Score :", grid_clf.best_score_)

    best_model_clf = grid_clf.best_estimator_
    hist_clf = best_model_clf.loss_history

    y_preds_test_clf = best_model_clf.predict(X_test_clf)
    y_preds_train_clf = best_model_clf.predict(X_train_clf)

    test_metrics_clf = compute_metrics_clf(y_preds_test_clf, y_test_clf)
    train_metrics_clf = compute_metrics_clf(y_preds_train_clf, y_train_clf)

    print("Train Metrics Classification :", train_metrics_clf)
    print("Test Metrics Classification :", test_metrics_clf)

    plt.figure(figsize=(10, 6))
    plt.plot(hist_clf, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs (Classification)')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = sklearn.metrics.confusion_matrix(y_test_clf, y_preds_test_clf)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()