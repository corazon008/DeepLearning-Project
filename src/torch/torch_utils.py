import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple

device = "cpu"


def train(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, epochs: int = 1) -> List[float]:
    model.train()

    loss_history = []

    for t in range(epochs):
        print(f"Epoch {t + 1} : ", end="")

        total_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = model.compute_loss(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= len(dataloader)
        loss_history.append(total_loss)

        print(f"Loss: {total_loss:.4f}")

    return loss_history


def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0
    collected_preds = []
    collected_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)

            loss = model.compute_loss(preds, y)
            total_loss += loss.item()

            collected_preds.append(preds)
            collected_targets.append(y)

    preds = torch.cat(collected_preds)
    targets = torch.cat(collected_targets)

    metrics = model.compute_metrics(preds, targets)

    return total_loss / len(dataloader), metrics
