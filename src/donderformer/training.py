"""One-epoch train and validation loops for models that predict ``(batch, T, 7)`` logits."""
from __future__ import annotations

import torch


# predict -> calculate loss -> backpropagate -> update weights
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)

        # before processing the new batch, we clear the gradients from the previous batch.
        optimizer.zero_grad()

        logits = model(X)
        targets = Y.argmax(dim=2)

        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )

        # learn from loss; how did each model weight contribute to the loss? how should we change the weights to reduce the loss?
        loss.backward()
        # use gradients to update the model's weights
        optimizer.step()

        total_loss += loss.item()

    # returns the average loss across all batches in the epoch
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)
            targets = Y.argmax(dim=2)

            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader)
