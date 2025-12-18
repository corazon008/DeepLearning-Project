import torch
from torch import nn
from typing import List, Optional


class MyDropout:
    def __init__(self, dropout_rates: Optional[List[float]] = None):
        # avoid mutable default argument
        self.dropout_rates = dropout_rates if dropout_rates is not None else [0.0]
        self.index = 0

    def get(self):
        if self.index > len(self.dropout_rates) - 1:
            return 0
        rate = self.dropout_rates[self.index]
        self.index += 1
        return rate


class ModelBase(nn.Module):
    def __init__(self, nb_features: int, output_vars: int = 1, layers: int = 2, width: int = 512,
                 dropout_rates: Optional[List[float]] = None, loss_fn=nn.MSELoss, activation=nn.ReLU):
        super().__init__()
        dropout = MyDropout(dropout_rates)
        self.net = nn.Sequential()
        self.net.add_module("input_layer", nn.Linear(nb_features, width))
        for i in range(layers):
            self.net.add_module(f"hidden_layer_{i + 1}", nn.Linear(width, width))
            self.net.add_module(f"relu_{i + 1}", activation())
            d = dropout.get()
            if d > 0:
                self.net.add_module(f"dropout_{i + 1}", nn.Dropout(d))

        self.net.add_module("output_layer", nn.Linear(width, output_vars))
        # Accept either a loss class (callable) or an already-instantiated loss object.
        if callable(loss_fn):
            # instantiate
            try:
                self.loss_fn = loss_fn()
            except Exception:
                # fallback: if instantiation fails, keep the object as-is
                self.loss_fn = loss_fn
        else:
            self.loss_fn = loss_fn

    def forward(self, x):
        return self.net(x)  # .squeeze(1)

    def compute_loss(self, preds, targets):
        return self.loss_fn(preds, targets)

    def compute_metrics(self, preds, targets):
        raise NotImplementedError


class RegressionModel(ModelBase):
    def __init__(self, nb_features: int, output_vars: int = 1, layers: int = 2, width: int = 512,
                 dropout_rates: Optional[List[float]] = None, loss_fn=nn.MSELoss, activation=nn.ReLU):
        super().__init__(nb_features=nb_features,
                         output_vars=output_vars,
                         layers=layers,
                         width=width,
                         dropout_rates=dropout_rates,
                         loss_fn=loss_fn,
                         activation=activation)


    def compute_metrics(self, preds, targets):
        # R² et MAE par exemple
        mae = torch.mean(torch.abs(preds - targets)).item()
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return {"MAE": mae, "R2": r2.item()}


class ClassificationModel(ModelBase):
    def __init__(self, nb_features: int, output_vars: int = 1, layers: int = 2, width: int = 512,
                 dropout_rates: Optional[List[float]] = None, loss_fn=nn.CrossEntropyLoss, activation=nn.ReLU):
        super().__init__(nb_features=nb_features,
                         output_vars=output_vars,
                         layers=layers,
                         width=width,
                         dropout_rates=dropout_rates,
                         loss_fn=loss_fn,
                         activation=activation)


    def compute_metrics(self, preds, targets):
        predicted = preds.argmax(1)
        accuracy = (predicted == targets).float().mean().item()
        return {"accuracy": accuracy}
