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
from data_proc import StandardScaleNorm, MinMaxNorm
import json
matplotlib.use('Agg')


def compute_stats(model_output):
    """Computes mean, median, 90th percentile, and 10th percentile
    for a given number of samples from the Bayesian Transformer model

    model_output: torch.Tensor [batch_size, pred_len, num_targets, num_samples]
    """

    mean = torch.mean(model_output, dim=-1)
    median, _ = torch.median(model_output, dim=-1)

    p90, p10 = torch.quantile(model_output, 0.9, dim=-1), torch.quantile(
        model_output, 0.1, dim=-1)

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
        plot_name):

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    model = torch.load(model_path, map_location=device)
    model.eval()

    # with open('modelsave/bmdet/best_hyperparams_non_spatial.json', 'r') as f:
    #     hyperparams = json.load(f)

    # model = BSMDeTWrapper(**hyperparams)
    # model.eval()

    test_loader = torch.load(test_loader_path)

    train_norm = torch.load(train_norm_path)[0]
    # print(train_norm)
    # if len(train_norm) == 1:
    #     train_norm = train_norm[0]

    results = []
    metrics = []
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        print(inputs.shape)
        outputs = model.test(
            in_test=inputs, samples=num_samples, scaler=train_norm)
        print(outputs.shape)
        mean, median, p90, p10 = compute_stats(outputs)
        metric = compute_metrics(forecasts=outputs, ground_truth=targets)
        metrics.append(metric)

        batch_size = mean.shape[0]
        predict_len = mean.shape[1]
        for i in range(batch_size):
            for t in range(predict_len):
                results.append({
                    'model_p10': p10[i, t, 0].item(),
                    'model_mean': mean[i, t, 0].item(),
                    'model_median': median[i, t, 0].item(),
                    'model_p90': p90[i, t, 0].item()
                })
        # Optionally, plot predictions for the first example in the first batch
        if batch_idx == 0:
            sample_idx = 0
            pred_mean = mean[sample_idx, :, 0].cpu().detach().numpy()
            pred_p90 = p90[sample_idx, :, 0].cpu().detach().numpy()
            pred_p10 = p10[sample_idx, :, 0].cpu().detach().numpy()
            true_vals = targets[sample_idx, :].cpu().detach().numpy()

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

    # Create a datetime index based on the raw test data.
    # Each day in the test period has 24 data points (hours 0-23).
    test_start_date = pd.to_datetime(test_start_date)
    test_end_date = pd.to_datetime(test_end_date)
    # Calculate expected total number of hours (inclusive of both endpoints)
    expected_hours = ((test_end_date - test_start_date).days + 1) * 24
    print("results len", len(results))
    if len(results) != expected_hours:
        print(
            f"Warning: Number of predictions ({len(results)}) does not match expected hours ({expected_hours}).")

    datetime_index = pd.date_range(
        start=test_start_date, periods=len(results), freq='h')

    df = pd.DataFrame(results, index=datetime_index)
    df.index.name = "datetime"

    df.to_csv(new_csv_savename)

    return {
        "ACR": np.mean([m['avg_coverage_rate'] for m in metrics]),
        "AIL": np.mean([m['avg_interval_length'] for m in metrics]),
        "AES": np.mean([m['energy_score'] for m in metrics])
    }


if __name__ == "__main__":
    res = main(model_path='modelsave/bmdet/bmdet_spatial.pt',
               test_loader_path='data/spatial/test_loader_spatial.pt',
               train_norm_path='data/spatial/transforms_spatial.pt',
               raw_csv_path='',
               num_samples=20,
               new_csv_savename='bmdet_output_stats.csv',
               test_start_date=datetime(2024, 9, 1),
               test_end_date=datetime(2025, 1, 6),
               device='cpu',
               plot_name='spatial')

    print(res)

    res = main(model_path='modelsave/bmdet/bmdet_non_spatial.pt',
               test_loader_path='data/non_spatial/test_loader_non_spatial.pt',
               train_norm_path='data/non_spatial/transforms_non_spatial.pt',
               raw_csv_path='',
               num_samples=20,
               new_csv_savename='bmdet_output_stats.csv',
               test_start_date=datetime(2024, 9, 1),
               test_end_date=datetime(2025, 1, 6),
               device='cpu',
               plot_name='non_spatial'
               )

    print(res)
