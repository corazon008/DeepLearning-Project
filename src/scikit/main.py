import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import(accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score)

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    RANDOM_STATE = 42


    df = pd.read_csv('../../data/health_lifestyle_dataset.csv')

    df.head()

    df.info()
    df.describe()

    print(f"\n\nClassification : -----------------------------")

    TARGET = 'disease_risk'
    print(f"On cherche à prédire s'il y a un risque de maladie (colonne {TARGET}).")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    print("Variables numériques :", numeric_features.tolist())
    print("Variables catégorielles :", categorical_features.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if y.nunique() < 20 else None
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("mlp", mlp)
    ])

    pipeline.fit(X_train, y_train)

    print("\nNombre d'itérations :", pipeline.named_steps["mlp"].n_iter_)

    y_pred = pipeline.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_pred = pipeline.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print(f"Matrice de confusion")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    print(f"\nOptimisation (GridSearchCV) :")

    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "mlp__alpha": [1e-5, 1e-4, 1e-3],
        "mlp__learning_rate_init": [1e-4, 1e-3]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score CV :", grid.best_score_)

    best_model = grid.best_estimator_

    y_pred_best = best_model.predict(X_test)

    print("\nAccuracy (optimisé) :", accuracy_score(y_test, y_pred_best))
    print(classification_report(y_test, y_pred_best))

    print(f"\nRégression linéaire : -----------------------------")
    print(f"On cherche à prédire le taux de cholesterol et les calories consomées (les colonnes 'cholesterol' et 'calories_consumed').")

    TARGETS = ["calories_consumed", "cholesterol"]

    X = df.drop(columns=TARGETS)
    Y = df[TARGETS]

    print(df.columns.tolist())

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), X.select_dtypes(include=["int64", "float64"]).columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes(include=["object"]).columns)
        ]
    )

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("mlp", mlp)
    ])

    pipeline.fit(X_train, Y_train)
    print("\nItérations :", pipeline.named_steps["mlp"].n_iter_)

    Y_pred = pipeline.predict(X_test)

    for i, target in enumerate(TARGETS):
        mse = mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])

        print(f" {target}")
        print(f"  MSE  : {mse:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R²   : {r2:.4f}")
        print()

    for i, target in enumerate(TARGETS):
        plt.figure(figsize=(4, 4))
        plt.scatter(Y_test.iloc[:, i], Y_pred[:, i], alpha=0.6)
        plt.plot(
            [Y_test.iloc[:, i].min(), Y_test.iloc[:, i].max()],
            [Y_test.iloc[:, i].min(), Y_test.iloc[:, i].max()],
            "r--"
        )
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Valeurs prédites")
        plt.title(f"Régression – {target}")
        plt.show()

