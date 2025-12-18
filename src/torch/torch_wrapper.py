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
                 epochs=20,
                 verbose=0,
                 early_stopping=True,
                 patience=10,
                 min_delta=1e-6):

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
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

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
            # ensure 1D
            if y.dim() > 1:
                y = y.view(-1)
            uniq = torch.unique(y)
            self.output_vars = len(uniq)
            if self.verbose:
                print(f"[fit] classifier detected. n_samples={X.shape[0]}, n_features={X.shape[1]}, classes={uniq.tolist()}")
            if self.output_vars < 2:
                print(f"[fit] WARNING: only one class present in training data: {uniq.tolist()}. The model cannot learn a meaningful classifier.")
        else:
            y = torch.tensor(y, dtype=torch.float32)
            self.output_vars = y.shape[1] if len(y.shape) > 1 else 1

        self.nb_features = X.shape[1]

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._build_model()
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

        best_loss = np.inf
        best_state = None
        epochs_no_improve = 0

        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                preds = self.model(batch_X)

                # debug: print shapes on first batch of first epoch
                if self.verbose and epoch == 0 and batch_idx == 0:
                    print(f"[train] batch shapes: X={batch_X.shape}, y={batch_y.shape}, preds={preds.shape}")

                loss = self.model.compute_loss(preds, batch_y)
                if torch.isnan(loss):
                    raise ValueError("NaN loss detected")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_loss = total_loss / len(dataloader)
            self.loss_history.append(epoch_loss)
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {self.loss_history[-1]:.4f}")

            # ----- EARLY STOPPING -----
            if self.early_stopping:
                if epoch_loss < best_loss - self.min_delta:
                    best_loss = epoch_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(best loss: {best_loss:.6f})"
                        )
                    break

        return self

    def predict(self, X, threshold=0.5):
        if self.model is None:
            raise ValueError("Model not fitted yet")

        self.model.eval()
        with torch.no_grad():
            if self.is_classifier:
                probs = self.predict_proba(X)
                # ensure shape (N,2) for binary
                if probs.ndim == 1:
                    probs = np.vstack([1 - probs, probs]).T
                preds = (probs[:, 1] >= threshold).astype(int)
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
            # Handle different shapes for binary classification:
            # - if model outputs single logit per sample: shape (N, 1) or (N,)
            # - if model outputs two logits per sample: shape (N, 2)
            if logits.dim() == 1:
                # shape (N,) -> treat as single logit for positive class
                pos_logits = logits.unsqueeze(1)
                neg_logits = torch.zeros_like(pos_logits)
                logits = torch.cat([neg_logits, pos_logits], dim=1)
            elif logits.dim() == 2 and logits.shape[1] == 1:
                # (N,1) single logit -> create two-class logits [0, logit]
                pos_logits = logits
                neg_logits = torch.zeros_like(pos_logits)
                logits = torch.cat([neg_logits, pos_logits], dim=1)

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
        # Instantiate loss_fn correctly whether it's a class or an instance
        if isinstance(self.loss_fn, type) and issubclass(self.loss_fn, nn.Module):
            loss = self.loss_fn()
        else:
            # try to use as-is; if it's callable but not a class, attempt to call and fallback
            try:
                loss = self.loss_fn() if callable(self.loss_fn) else self.loss_fn
            except Exception:
                loss = self.loss_fn

        return RegressionModel(
            nb_features=self.nb_features,
            output_vars=self.output_vars,
            width=self.width,
            layers=self.layers,
            activation=self.activation,
            loss_fn=loss,
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
        # loss_fn in wrapper may be a class or a instantiated loss
        if isinstance(self.loss_fn, type) and issubclass(self.loss_fn, nn.Module):
            # instantiate with weight if provided
            if self.class_weights is not None:
                loss = self.loss_fn(weight=self.class_weights.to(device))
            else:
                loss = self.loss_fn()
        else:
            # assume it's already an instance; if class_weights provided and loss supports weight, try to set
            loss = self.loss_fn
            if self.class_weights is not None and hasattr(loss, 'weight'):
                try:
                    loss.weight = self.class_weights.to(device)
                except Exception:
                    pass

        return ClassificationModel(
            nb_features=self.nb_features,
            output_vars=self.output_vars,
            width=self.width,
            layers=self.layers,
            activation=self.activation,
            loss_fn=loss,
            dropout_rates=self.dropout_rates
        ).to(device)
