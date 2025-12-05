import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import List


class MyDropout:
    def __init__(self, dropout_rates: List[float] = [0]):
        self.dropout_rates = dropout_rates
        self.index = 0

    def get(self):
        if self.index > len(self.dropout_rates) - 1:
            return 0
        rate = self.dropout_rates[self.index]
        self.index += 1
        return rate


class RegressionModel(nn.Module):
    def __init__(self, nb_features: int, layers:int=2, width: int = 512, dropout_rates: List[float] = [0.0], loss_fn=nn.MSELoss(), activation=nn.ReLU()):
        super().__init__()
        dropout = MyDropout(dropout_rates)
        self.net = nn.Sequential()
        self.net.add_module("input_layer", nn.Linear(nb_features, width))
        for i in range(layers):
            self.net.add_module(f"hidden_layer_{i+1}", nn.Linear(width, width))
            self.net.add_module(f"relu_{i+1}",activation)
            self.net.add_module(f"dropout_{i+1}", nn.Dropout(dropout.get()))

        self.net.add_module("output_layer", nn.Linear(width, 1))
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.net(x).squeeze(1)

    def compute_loss(self, preds, targets):
        return self.loss_fn(preds, targets)

    def compute_metrics(self, preds, targets):
        # R² et MAE par exemple
        mae = torch.mean(torch.abs(preds - targets)).item()
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return {"MAE": mae, "R2": r2.item()}


class ClassificationModel(nn.Module):
    def __init__(self, nb_features: int, num_classes: int, loss_fn=nn.CrossEntropyLoss(), width: int = 512, dropout_rates: List[float] = [0.0]):
        super().__init__()
        dropout = MyDropout(dropout_rates)
        self.net = nn.Sequential(
            nn.Linear(nb_features, width), nn.ReLU(),
            nn.Dropout(dropout.get()),
            nn.Linear(width, width), nn.ReLU(),
            nn.Dropout(dropout.get()),
            nn.Linear(width, 1)
        )
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.net(x)  # logits

    def compute_loss(self, preds, targets):
        return self.loss_fn(preds, targets)

    def compute_metrics(self, preds, targets):
        predicted = preds.argmax(1)
        accuracy = (predicted == targets).float().mean().item()
        return {"accuracy": accuracy}
