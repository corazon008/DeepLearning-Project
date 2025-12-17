from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import numpy as np

from src.torch.torch_model import RegressionModel, ClassificationModel

device = "cpu"


class Skwrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 width=128,
                 layers=2,
                 dropout_rates=None,
                 lr=1e-3,
                 activation: nn.Module = nn.ReLU,
                 optimizer: Callable = optim.Adam,
                 loss_fn: Callable = nn.MSELoss,
                 batch_size=32,
                 epochs=10,
                 verbose=0):

        self.width = width
        self.layers = layers
        self.dropout_rates = dropout_rates if dropout_rates is not None else [0.0]
        self.lr = lr
        self.activation = activation
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.is_classifier = False  # to be set in subclass if needed
        self.model = None
        self.loss_history = []

    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        # convert target dtype depending on task
        if self.is_classifier:
            # CrossEntropyLoss expects targets as long (class indices) with shape (N,)
            y = torch.tensor(y, dtype=torch.long).squeeze()
            self.output_vars = len(torch.unique(y))
        else:
            y = torch.tensor(y, dtype=torch.float32)
            self.output_vars = y.shape[1] if len(y.shape) > 1 else 1

        self.nb_features = X.shape[1]

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._build_model()
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                preds = self.model(batch_X)

                loss = self.model.compute_loss(preds, batch_y)
                if torch.isnan(loss):
                    raise ValueError("NaN loss detected")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.loss_history.append(total_loss / len(dataloader))
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {self.loss_history[-1]:.4f}")

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")

        self.model.eval()
        with torch.no_grad():
            if self.is_classifier:
                probs = self.predict_proba(X)
                preds = probs.argmax(axis=1)
                return preds
            else:
                X = torch.tensor(X, dtype=torch.float32).to(device)
                preds = self.model(X)
        return preds.cpu().numpy().ravel().reshape(-1, self.output_vars)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


class TorchRegressor(Skwrapper):
    def __init__(self,
                 width=128,
                 layers=2,
                 dropout_rates=None,
                 lr=1e-3,
                 activation: nn.Module = nn.ReLU,
                 optimizer: Callable = optim.Adam,
                 loss_fn: Callable = nn.MSELoss,
                 batch_size=32,
                 epochs=10, ):

        super().__init__(
            width=width,
            layers=layers,
            dropout_rates=dropout_rates,
            lr=lr,
            optimizer=optimizer,
            activation=activation,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs=epochs
        )

    def _build_model(self):
        return RegressionModel(
            nb_features=self.nb_features,
            output_vars=self.output_vars,
            width=self.width,
            layers=self.layers,
            activation=self.activation,
            loss_fn=self.loss_fn(),
            dropout_rates=self.dropout_rates
        ).to(device)



class TorchClassifier(Skwrapper):
    def __init__(self,
                 width=128,
                 layers=2,
                 dropout_rates=None,
                 lr=1e-3,
                 activation: nn.Module = nn.ReLU,
                 optimizer: Callable = optim.Adam,
                 loss_fn: Callable = nn.CrossEntropyLoss,
                 batch_size=32,
                 epochs=10,
                 class_weights=None):

        self.class_weights = class_weights

        super().__init__(
            width=width,
            layers=layers,
            dropout_rates=dropout_rates,
            lr=lr,
            optimizer=optimizer,
            activation=activation,
            loss_fn=loss_fn,
            batch_size=batch_size,
            epochs=epochs,
        )
        # mark this wrapper as a classifier so it handles target dtype and predict
        self.is_classifier = True

    def _build_model(self):
        if self.class_weights is not None:
            loss = self.loss_fn(weight=self.class_weights.to(device))
        else:
            loss = self.loss_fn()

        return ClassificationModel(
            nb_features=self.nb_features,
            output_vars=self.output_vars,
            width=self.width,
            layers=self.layers,
            activation=self.activation,
            loss_fn=loss,
            dropout_rates=self.dropout_rates
        ).to(device)
