import os
import torch
import itertools
import argparse
import matplotlib.pyplot as plt
import numpy as np

from deepar.model import DeepAR
from deepar.trainer import DeepARTrainer, grid_search
from data_proc import StandardScaleNorm, MinMaxNorm, TransformSequence


def main(spatial='spatial', dataset="spain_data", savename="_1", grid_search_epochs=1, extra_epochs=50):
    spatial = True if spatial == 'spatial' else False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    suffix = "spatial_AR" if spatial else "non_spatial_AR"

    transforms = torch.load(os.path.join(
        f"data/{dataset}_{suffix}", f"transforms_{suffix}.pt"))
    print(f"Loaded data transforms: {type(transforms)}")

    transforms.set_device(device)
    print(f"Set transforms to device: {device}")

    train_loader = torch.load(os.path.join(
        f"data/{dataset}_{suffix}", f"train_loader_{suffix}.pt"))
    val_loader = torch.load(os.path.join(
        f"data/{dataset}_{suffix}", f"val_loader_{suffix}.pt"))
    test_loader = torch.load(os.path.join(
        f"data/{dataset}_{suffix}", f"test_loader_{suffix}.pt"))

    sample_batch = next(iter(train_loader))
    print(
        f"Train sample shapes: target={sample_batch[0].shape}, covariates={sample_batch[1].shape}, mask={sample_batch[2].shape}")

    covariate_dim = sample_batch[1].shape[-1]
    print(f"Covariate dimension: {covariate_dim}")

    hyperparameter_grid = {
        "num_class": [100],  # Number of distinct time series identifiers
        "covariate_size": [covariate_dim],
        "hidden_size": [40],
        "num_layers": [3],
        "embedding_dim": [32],
        "learning_rate": [1e-3],
        "predict_steps": [24],  # Number of steps to forecast
        "predict_start": [336]  # Index where forecasting starts
    }

    print(f"Starting grid search with hyperparameters: {hyperparameter_grid}")

    best_model, best_config, best_loss, results = grid_search(
        hyperparameter_grid,
        train_loader,
        val_loader,
        device=device,
        data_norm=transforms,
        num_epochs=grid_search_epochs,
        extra_epochs=extra_epochs,
        savename=f"best_deepar_model_{suffix}{savename}"
    )

    print(f"Grid search completed. Best validation loss: {best_loss:.4f}")
    print(f"Best configuration: {best_config}")

    print("Generating forecasts...")
    samples, means, sigmas = best_model.forecast(
        test_loader=test_loader,
        num_samples=40,
        data_norm=transforms
    )
    samples = samples.squeeze(-1)
    print('samples shape', samples.shape)

    first_batch = next(iter(test_loader))
    context_target, _, mask = first_batch

    forecast_start = mask[0].sum().item()

    plot_batches = [1, 32, 63]

    for plot_batch in plot_batches:
        true_values_all = context_target[plot_batch, :].cpu().numpy()
        true_values_forecast = context_target[plot_batch, forecast_start:].cpu(
        ).numpy()
        forecast_samples = samples[:, plot_batch, :].cpu().numpy()
        forecast_mean = means[plot_batch, :].squeeze(-1).cpu().numpy()
        forecast_sigma = sigmas[plot_batch, :].squeeze(-1).cpu().numpy()

        plt.figure(figsize=(15, 6))

        plt.plot(range(len(true_values_all)), true_values_all,
                 'k-', linewidth=2, label='True Values')

        plt.axvline(x=forecast_start, color='gray',
                    linestyle='--', label='Forecast Start')

        num_samples_to_plot = min(20, forecast_samples.shape[0])
        for i in range(num_samples_to_plot):
            plt.plot(range(forecast_start, forecast_start + len(forecast_mean)),
                     forecast_samples[i, :], 'b-', alpha=0.3)

        plt.plot(range(forecast_start, forecast_start + len(forecast_mean)),
                 forecast_mean, 'r-', linewidth=2, label='Mean Forecast')

        plt.fill_between(
            range(forecast_start, forecast_start + len(forecast_mean)),
            forecast_mean - 2 * forecast_sigma,
            forecast_mean + 2 * forecast_sigma,
            color='r', alpha=0.2, label='95% Confidence Interval'
        )

        plt.title(f'DeepAR Forecast vs True Values ({suffix})')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        plt.savefig(f'deepar_forecast_{suffix}_{plot_batch}.png')
        plt.close()

    print(f"Forecast plot saved as deepar_forecast_{suffix}.png")

    print("\nGrid search results summary:")
    for config, loss in results:
        print(f"Config: {config}, Val Loss: {loss:.4f}")
    print(f"Best Config: {best_config} with loss {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial', type=str, default='non_spatial',
                        help='Use spatial features ("spatial") or non-spatial features ("non_spatial")')
    parser.add_argument('--dataset', type=str, default='spain_data',
                        help='Use spain_data or ercot_data')
    parser.add_argument('--savename', type=str, default='',
                        help='suffix for savename of best model -- differentiates between different runs')
    parser.add_argument('--grid_search_epochs', type=int, default=1,
                        help='number of epochs for grid search')
    parser.add_argument('--extra_epochs', type=int, default=50,
                        help='number of extra epochs for training')
    args = parser.parse_args()
    main(spatial=args.spatial, dataset=args.dataset, savename=args.savename,
         grid_search_epochs=args.grid_search_epochs, extra_epochs=args.extra_epochs)
