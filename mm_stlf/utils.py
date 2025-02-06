import torch
import torch.nn as nn
import numpy as np


def mse_loss(predictions, targets):
    """Computes Mean Squared Error (MSE) loss."""
    return nn.MSELoss()(predictions, targets)


def pinball_loss(predictions, targets, quantiles=[0.1, 0.5, 0.9]):
    """
    Computes Pinball loss for quantile regression.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.
        quantiles (list): List of quantiles for forecasting.

    Returns:
        torch.Tensor: Pinball loss.
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - predictions[..., i]
        loss = torch.max(q * errors, (q - 1) * errors).mean()
        losses.append(loss)
    return sum(losses) / len(quantiles)


def rmse(predictions, targets):
    """Computes Root Mean Squared Error (RMSE)."""
    return torch.sqrt(mse_loss(predictions, targets))


def mae(predictions, targets):
    """Computes Mean Absolute Error (MAE)."""
    return torch.mean(torch.abs(predictions - targets))


def evaluate_metrics(predictions, targets):
    """Computes various evaluation metrics."""
    predictions, targets = predictions.detach().cpu(), targets.detach().cpu()
    return {
        "RMSE": rmse(predictions, targets).item(),
        "MAE": mae(predictions, targets).item(),
    }
