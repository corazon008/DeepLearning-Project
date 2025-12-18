import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scikeras.wrappers import KerasRegressor, KerasClassifier
from sklearn.model_selection import GridSearchCV

import os
import sys
sys.path.append(os.path.abspath(os.path.join('..').join("..")))
from src.tf.tf_wrapper import *


if __name__ == "__main__":

    df = pd.read_csv("../../data/health_lifestyle_dataset_cleaned.csv")

    print(f"REGRESSION ---------------")
    regression_target = ["cholesterol", "calories_consumed"]
    features_reg = df.drop(columns=regression_target).values
    regression_labels = df[regression_target].values
    print(f"Les colonnes que nous cherchons à prédire sont : {regression_target}")

    scaler_y = StandardScaler()
    scaler_y.fit(df[regression_target])

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(features_reg, regression_labels, test_size=0.2, random_state=42)

    model_reg = KerasRegressor(
        model=build_tf_regressor,
        nb_features=X_train_reg.shape[1],
        layers_count=2,
        width=64,
        activation="relu",
        dropout_rate=0.0,
        learning_rate=1e-3,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    param_grid_reg = {
        "model__layers_count": [2, 3],
        "model__width": [64, 128],
        "model__activation": ["relu", "tanh"],
        "model__dropout_rate": [0.0, 0.2],
        "model__learning_rate": [1e-1, 1e-3],
        "epochs": [30],
        "batch_size": [32]
    }
    print(f"Voici les paramètres que allons tester pour la régression : {param_grid_reg}")
    print(f"\nDébut de l\'entraînement avec GridSearchCV...")

    grid = GridSearchCV(estimator=model_reg, param_grid=param_grid_reg, cv=3, scoring="r2", n_jobs=-1, verbose=2)
    grid_result_reg = grid.fit(X_train_reg, y_train_reg)

    print(f"Fin de l\'entraînement")
    print(f"\nMeilleur score : {grid_result_reg.best_score_}")
    print(f"Meilleurs paramètres : {grid_result_reg.best_params_}")

    best_model_reg = grid_result_reg.best_estimator_

    y_pred_scaled = best_model_reg.predict(X_train_reg)
    y_pred_final = scaler_y.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_train_reg, y_pred_final)
    mae = mean_absolute_error(y_train_reg, y_pred_final)
    r2 = r2_score(y_train_reg, y_pred_final)

    print(f"\nPerformance du meilleur modèle de régression sur les données d\'entraînement :")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    y_pred_scaled = best_model_reg.predict(X_test_reg)
    y_pred_final = scaler_y.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_test_reg, y_pred_final)
    mae = mean_absolute_error(y_test_reg, y_pred_final)
    r2 = r2_score(y_test_reg, y_pred_final)

    print(f"Performance du meilleur modèle de régression sur les données de test :")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    print(f"\n\nCLASSIFICATION ---------------")
    classification_target = "disease_risk"
    features_clas = df.drop(columns=classification_target).values
    classification_labels = df[classification_target].values
    print(f"La colonne que nous cherchons à prédire est : {classification_target}")

    X_train_clas, X_test_clas, y_train_clas, y_test_clas = train_test_split(features_clas, classification_labels, test_size=0.2, random_state=42)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model_clas = KerasClassifier(
        model=build_tf_classifier,
        nb_features=X_train_clas.shape[1],
        layers_count=2,
        width=64,
        activation='relu',
        dropout_rate=0.0,
        learning_rate=1e-3,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    param_grid_clas = {
        "model__layers_count": [2, 3],
        "model__width": [64, 128],
        "model__activation": ["relu", "tanh"],
        "model__dropout_rate": [0.0, 0.2],
        "model__learning_rate": [1e-1, 1e-3],
        "epochs": [30],
        "batch_size": [32]
    }
    print(f"Voici les paramètres que allons tester pour la classification : {param_grid_clas}")
    print(f"\nDébut de l\'entraînement avec GridSearchCV...")

    grid_clas = GridSearchCV(estimator=model_clas, param_grid=param_grid_clas, cv=skf, scoring="accuracy", n_jobs=-1, verbose=2)
    grid_clas_result = grid_clas.fit(X_train_clas, y_train_clas)

    print(f"Fin de l\'entraînement")
    print(f"\nMeilleure Accuracy : {grid_clas_result.best_score_:.4f}")
    print(f"Meilleurs paramètres : {grid_clas_result.best_params_}")

    best_model_clas = grid_clas_result.best_estimator_

    print(f"\nPerformance du meilleur modèle de régression sur les données d\'entraînement :")
    y_pred = best_model_clas.predict(X_train_clas)
    f1 = f1_score(y_train_clas, y_pred, average='weighted')
    accuracy = accuracy_score(y_train_clas, y_pred)
    print(f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

    print(f"Performance du meilleur modèle de régression sur les données de test :")
    y_pred = best_model_clas.predict(X_test_clas)
    f1 = f1_score(y_test_clas, y_pred, average='weighted')
    accuracy = accuracy_score(y_test_clas, y_pred)
    print(f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

    print(classification_report(y_test_clas, y_pred))

    sns.heatmap(confusion_matrix(y_test_clas, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion")
    plt.show()