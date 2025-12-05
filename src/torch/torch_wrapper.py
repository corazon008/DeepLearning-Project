from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim

from src.torch.torch_model import RegressionModel

device = "cpu"


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 nb_features=None,
                 width=128,
                 dropout_rates=[0.0],
                 lr=1e-3,
                 batch_size=32,
                 epochs=10, ):

        self.nb_features = nb_features
        self.width = width
        self.dropout_rates = dropout_rates
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.model = None

    def _build_model(self):
        return RegressionModel(
            nb_features=self.nb_features,
            width=self.width,
            dropout_rates=self.dropout_rates
        ).to(self.device)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.nb_features is None:
            self.nb_features = X.shape[1]

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._build_model()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                preds = self.model(batch_X)
                loss = self.model.compute_loss(preds, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X)
        return preds.cpu().numpy().ravel()
