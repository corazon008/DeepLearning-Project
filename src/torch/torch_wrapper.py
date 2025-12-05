from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim

from src.torch.torch_model import RegressionModel

device = "cpu"


class Skwrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 nb_features=None,
                 width=128,
                 layers=2,
                 dropout_rates=[0.0],
                 lr=1e-3,
                 activation: nn.Module = nn.ReLU(),
                 optimizer: optim.Optimizer = optim.Adam,
                 loss_fn: nn.Module = nn.MSELoss(),
                 batch_size=32,
                 epochs=10, ):

        self.nb_features = nb_features
        self.width = width
        self.layers = layers
        self.dropout_rates = dropout_rates
        self.lr = lr
        self.activation = activation
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = None

    def _build_model(self):
        raise NotImplementedError

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.nb_features is None:
            self.nb_features = X.shape[1]

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._build_model()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                preds = self.model(batch_X)
                loss = self.model.compute_loss(preds, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(device)
            preds = self.model(X)
        return preds.cpu().numpy().ravel()


class TorchRegressor(Skwrapper):
    def __init__(self,
                 nb_features=None,
                 width=128,
                 layers=2,
                 dropout_rates=[0.0],
                 lr=1e-3,
                 activation: nn.Module = nn.ReLU(),
                 optimizer: optim.Optimizer = optim.Adam,
                 loss_fn: nn.Module = nn.MSELoss(),
                 batch_size=32,
                 epochs=10, ):

        super().__init__(
            nb_features=nb_features,
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
            width=self.width,
            layers=self.layers,
            activation=self.activation,
            loss_fn=self.loss_fn,
            dropout_rates=self.dropout_rates
        ).to(device)
