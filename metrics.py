import numpy as np
import torch


def compute_acr_ail(forecasts, ground_truth, alpha=0.2):
    # average coverage rate and average interval length
    lower_bound = np.percentile(forecasts, 100 * (alpha / 2), axis=-1)
    upper_bound = np.percentile(forecasts, 100 * (1 - alpha / 2), axis=-1)

    interval_length = upper_bound - lower_bound
    in_interval = (ground_truth >= lower_bound) & (ground_truth <= upper_bound)
    return np.mean(in_interval), np.mean(interval_length)


def compute_aes(forecasts, ground_truth):
    # average energy score
    batch_size, num_hours, num_variates, num_samples = forecasts.shape
    forecasts_flat = forecasts.reshape(-1, num_samples)
    ground_truth_flat = ground_truth.reshape(-1, 1)

    term1 = np.mean(np.linalg.norm(forecasts_flat - ground_truth_flat, axis=1))

    # ||F - F'||_2 (mean pairwise distance between forecast samples)
    pairwise_diffs = np.sqrt(
        np.sum((forecasts_flat[:, :, None] -
               forecasts_flat[:, None, :]) ** 2, axis=1)
    )
    term2 = np.mean(pairwise_diffs)

    return term1 - 0.5 * term2


def compute_metrics(forecasts, ground_truth, alpha=0.2):
    """
    Compute metrics for probabilistic forecasts:
    - 80% confidence interval
    - Average coverage rate
    - Average interval length
    - Energy score

    Args:
        forecasts (torch.Tensor or np.ndarray): Forecasts of shape [batch_size, num_hours, num_variates, num_samples].
        ground_truth (torch.Tensor or np.ndarray): Ground truth of shape [batch_size, num_hours, num_variates].
        alpha (float): Significance level for the confidence interval (default is 0.2 for 80%).

    Returns:
        dict: Dictionary containing avg_coverage_rate, avg_interval_length, and energy_score.
    """
    if isinstance(forecasts, torch.Tensor):
        forecasts = forecasts.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.unsqueeze(-1)
        # ground_truth = train_scaler.reverse(ground_truth)
        ground_truth = ground_truth.detach().cpu().numpy()
    if forecasts.shape[:-1] != ground_truth.shape:
        raise ValueError(f"Forecasts shape {
                         forecasts.shape[:-1]} and ground truth shape {ground_truth.shape} are not aligned.")

    avg_coverage_rate, avg_interval_length = compute_acr_ail(
        forecasts, ground_truth, alpha)
    energy_score = compute_aes(forecasts, ground_truth)

    # Return results
    return {
        "avg_coverage_rate": avg_coverage_rate,
        "avg_interval_length": avg_interval_length,
        "energy_score": energy_score
    }
