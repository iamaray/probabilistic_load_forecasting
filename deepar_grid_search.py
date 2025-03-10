import os
import torch
import itertools
import argparse
import matplotlib.pyplot as plt
import numpy as np

from deepar.model import DeepAR
from deepar.trainer import DeepARTrainer, grid_search
from data_proc import StandardScaleNorm, MinMaxNorm, TransformSequence


def main(spatial='spatial'):
    spatial = True if spatial == 'spatial' else False
    # Set device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    suffix = "spatial_AR" if spatial else "non_spatial_AR"


    transforms = torch.load(os.path.join(
        "data", suffix, f"transforms_{suffix}.pt"))
    print(type(transforms))
    # Load the saved data loaders.
    train_loader = torch.load(os.path.join(
        "data", suffix, f"train_loader_{suffix}.pt"))
    val_loader = torch.load(os.path.join(
        "data", suffix, f"val_loader_{suffix}.pt"))

    for i, (x, y, z) in enumerate(train_loader):
        print(x.shape, y.shape, z.shape)

    test_loader = torch.load(os.path.join(
        "data", suffix, f"test_loader_{suffix}.pt"))

    sample = next(iter(train_loader))
    # sample[1] is the covariate tensor with shape: [batch_size, T, covariate_dim]
    covariate_dim = sample[1].shape[-1]

    # hyperparameter_grid = {
    #     "covariate_size": [covariate_dim],
    #     "hidden_size": [20, 40, 60],
    #     "num_layers": [1, 2, 3],
    #     "embedding_dim": [32, 64],
    #     "learning_rate": [1e-3]
    # }

    hyperparameter_grid = {
        "covariate_size": [covariate_dim],
        "hidden_size": [20],
        "num_layers": [1],
        "embedding_dim": [32],
        "learning_rate": [1e-3]
    }

    best_model, best_config, best_loss, results = grid_search(
        hyperparameter_grid,
        train_loader,
        val_loader,
        device=device,
        data_norm=transforms,
        num_epochs=5,
        savename="best_deepar_model"
    )

    samples, means, sigmas = best_model.forecast(
        test_loader=test_loader, num_samples=40)

    first_batch = next(iter(test_loader))

    context_target, _, mask = first_batch

    forecast_start = mask[0].sum().item()
    true_values = context_target[0, forecast_start:].cpu().numpy()

    forecast_samples = samples[0][:, 0, :].cpu().numpy()
    forecast_mean = means[0][0, :].cpu().numpy()
    forecast_sigma = sigmas[0][0, :].cpu().numpy()

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the true values
    plt.plot(true_values, 'k-', linewidth=2, label='True Values')

    # Plot a subset of samples for clarity (e.g., 10 samples)
    num_samples_to_plot = min(10, forecast_samples.shape[0])
    for i in range(num_samples_to_plot):
        plt.plot(forecast_samples[i], 'b-', alpha=0.3)

    # Plot the mean forecast
    plt.plot(forecast_mean, 'r-', linewidth=2, label='Mean Forecast')

    # Plot confidence intervals (mean ± 2*sigma)
    plt.fill_between(
        np.arange(len(forecast_mean)),
        forecast_mean - 2 * forecast_sigma,
        forecast_mean + 2 * forecast_sigma,
        color='r', alpha=0.2, label='95% Confidence Interval'
    )

    plt.title('DeepAR Forecast vs True Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f'deepar_forecast_{suffix}.png')
    plt.close()

    print(f"Forecast plot saved as deepar_forecast_{suffix}.png")

    print("Grid search results:")
    for config, loss in results:
        print(f"Config: {config}, Val Loss: {loss:.4f}")
    print(f"Best Config: {best_config} with loss {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial', type=str, default='non_spatial',
                        help='Use spatial features ("spatial") or non-spatial features ("non_spatial")')
    args = parser.parse_args()
    main(spatial=args.spatial)
