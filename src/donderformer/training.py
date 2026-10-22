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


# ---- event-token models (DonderFormer): loaders yield (mel, tokens, sep), see chart_dataset.collate

def train_epoch_tokens(model, loader, optimizer, device, scheduler=None, label_smoothing=0.02,
                       target_only=False, clip_norm=1.0, amp_dtype=None):
    """One epoch of teacher-forced next-token training; returns the mean loss."""
    from donderformer.models.donderformer import sequence_loss

    model.train()
    total_loss = 0.0
    for mel, tokens, sep in loader:
        mel, tokens = mel.to(device), tokens.to(device)
        optimizer.zero_grad()
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(mel, tokens)
        loss = sequence_loss(logits.float(), tokens, sep, label_smoothing, target_only)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_tokens(model, loader, device, target_only=True, amp_dtype=None):
    """Mean next-token loss (no label smoothing); by default only on target-region tokens."""
    from donderformer.models.donderformer import sequence_loss

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for mel, tokens, sep in loader:
            mel, tokens = mel.to(device), tokens.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(mel, tokens)
            total_loss += sequence_loss(logits.float(), tokens, sep, 0.0, target_only).item()
    return total_loss / len(loader)
