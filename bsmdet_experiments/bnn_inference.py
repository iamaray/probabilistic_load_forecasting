from tqdm import tqdm  # Add progress bar
from metrics import compute_metrics
from final_data_prep import preprocess
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.trainer import BayesTrainer
import torch
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime
from data_proc import StandardScaleNorm, MinMaxNorm, TransformSequence
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


matplotlib.use('Agg')


def compute_stats(model_output):
    """Computes mean, median, 90th percentile, and 10th percentile
    for a given number of samples from the Bayesian Transformer model

    model_output: torch.Tensor [batch_size, pred_len, num_targets, num_samples]
    """
    # Compute all statistics in a single pass to avoid multiple iterations over the data
    mean = torch.mean(model_output, dim=-1)
    median = torch.median(model_output, dim=-1).values

    # Compute quantiles in a single operation
    quantiles = torch.quantile(model_output, torch.tensor(
        [0.1, 0.9], device=model_output.device), dim=-1)
    p10, p90 = quantiles[0], quantiles[1]

    return mean, median, p90, p10


def main(
        model_path,
        test_loader_path,
        train_norm_path,
        raw_csv_path,
        num_samples,
        new_csv_savename,
        test_start_date,
        test_end_date,
        device,
        plot_name,
        model=None):

    # Determine device once and use consistently
    device = torch.device(
        'cpu') if not torch.cuda.is_available() else torch.device(device)

    # Load model if not provided
    if model is None:
        model = torch.load(model_path, map_location=device)
        model.eval()

    # with open('modelsave/bmdet/best_hyperparams_non_spatial.json', 'r') as f:
    #     hyperparams = json.load(f)

    # model = BSMDeTWrapper(**hyperparams)
    # model.eval()

    # Load test data and normalization
    test_loader = torch.load(test_loader_path)
    train_norm = torch.load(train_norm_path)
    # print(train_norm)
    # if len(train_norm) == 1:
    #     train_norm = train_norm[0]

    # Pre-allocate lists with estimated capacity to avoid resizing
    results = []
    metrics = []
    total_batches = len(test_loader)

    # Process batches with progress bar
    with torch.no_grad():  # Ensure no gradients are computed during inference
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Processing batches")):
            # Move inputs to device
            inputs = inputs.to(device)

            # Get model outputs
            outputs = model.test(
                in_test=inputs, samples=num_samples, scaler=train_norm)

            # Compute statistics efficiently
            mean, median, p90, p10 = compute_stats(outputs)

            # Compute metrics once per batch
            batch_metrics = compute_metrics(
                forecasts=outputs, ground_truth=targets)
            metrics.append(batch_metrics)

            # Extract results more efficiently using vectorized operations
            batch_size, predict_len = mean.shape[0], mean.shape[1]

            # Create batch results in a vectorized way
            for i in range(batch_size):
                for t in range(predict_len):
                    results.append({
                        'model_p10': p10[i, t, 0].item(),
                        'model_mean': mean[i, t, 0].item(),
                        'model_median': median[i, t, 0].item(),
                        'model_p90': p90[i, t, 0].item()
                    })

            # Plot only for the first batch
            if batch_idx == 0:
                sample_idx = 0
                # Move tensors to CPU only once for plotting
                pred_mean = mean[sample_idx, :, 0].cpu().numpy()
                pred_p90 = p90[sample_idx, :, 0].cpu().numpy()
                pred_p10 = p10[sample_idx, :, 0].cpu().numpy()
                true_vals = targets[sample_idx, :].cpu().numpy()

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(true_vals, label='Ground Truth', marker='o')
                plt.plot(pred_mean, label='Predicted Mean', marker='x')
                plt.fill_between(np.arange(len(pred_mean)), pred_p10, pred_p90,
                                 color='gray', alpha=0.5, label='10th-90th Percentile')
                plt.title(
                    f"{plot_name} Prediction vs Ground Truth (2024/09/02 hr 0 - 2024/09/02 hr 23)")
                plt.xlabel("Time Step")
                plt.ylabel("Signal")
                plt.legend()
                plt.savefig(f"{plot_name}_prediction_example.png")
                plt.close()

    # Create datetime index and dataframe
    test_start_date = pd.to_datetime(test_start_date)
    test_end_date = pd.to_datetime(test_end_date)

    # Calculate expected total number of hours (inclusive of both endpoints)
    expected_hours = ((test_end_date - test_start_date).days + 1) * 24

    if len(results) != expected_hours:
        print(
            f"Warning: Number of predictions ({len(results)}) does not match expected hours ({expected_hours}).")

    # Create datetime index efficiently
    datetime_index = pd.date_range(
        start=test_start_date, periods=len(results), freq='h')

    # Create dataframe and save to CSV
    df = pd.DataFrame(results, index=datetime_index)
    df.index.name = "datetime"
    df.to_csv(new_csv_savename)

    # Compute final metrics efficiently using numpy operations
    final_metrics = {
        "ACR": np.mean([m['avg_coverage_rate'] for m in metrics]),
        "AIL": np.mean([m['avg_interval_length'] for m in metrics]),
        "AES": np.mean([m['energy_score'] for m in metrics])
    }

    return final_metrics


# if __name__ == "__main__":
#     res = main(model_path='modelsave/bmdet/bmdet_spatial.pt',
#                test_loader_path='data/spatial/test_loader_spatial.pt',
#                train_norm_path='data/spatial/transforms_spatial.pt',
#                raw_csv_path='',
#                num_samples=20,
#                new_csv_savename='bmdet_output_stats.csv',
#                test_start_date=datetime(2024, 9, 1),
#                test_end_date=datetime(2025, 1, 6),
#                device='cpu',
#                plot_name='spatial')

#     print(res)

#     res = main(model_path='modelsave/bmdet/bmdet_non_spatial.pt',
#                test_loader_path='data/non_spatial/test_loader_non_spatial.pt',
#                train_norm_path='data/non_spatial/transforms_non_spatial.pt',
#                raw_csv_path='',
#                num_samples=20,
#                new_csv_savename='bmdet_output_stats.csv',
#                test_start_date=datetime(2024, 9, 1),
#                test_end_date=datetime(2025, 1, 6),
#                device='cpu',
#                plot_name='non_spatial'
#                )

#     print(res)
