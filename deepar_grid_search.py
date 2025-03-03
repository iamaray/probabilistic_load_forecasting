import os
import torch
import itertools
import argparse

from deepar.model import DeepAR
from deepar.trainer import DeepARTrainer, grid_search


def main(spatial='spatial'):
    spatial = True if spatial == 'spatial' else False
    # Set device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Run data processing with ar_model=True so that we get autoregressive data loaders.
    # (Your benchmark_preprocess() function saves the datasets and loaders with an appropriate suffix.)
    # For convenience, we now load the saved loaders from disk.
    # The suffix will be "spatial_AR" if spatial is True; otherwise "non_spatial_AR".
    suffix = "spatial_AR" if spatial else "non_spatial_AR"

    # First, call benchmark_preprocess() to process and save the data.
    # (Note: benchmark_preprocess() returns the fitted transforms; the loaders are saved on disk.)
    transforms = torch.load

    # Load the saved data loaders.
    train_loader = torch.load(os.path.join(
        "data", suffix, f"train_loader_{suffix}.pt"))
    val_loader = torch.load(os.path.join(
        "data", suffix, f"val_loader_{suffix}.pt"))

    for i, (x, y, z) in enumerate(train_loader):
        print(x.shape, y.shape, z.shape)

    # Optionally, you can load the test_loader as well.
    # test_loader  = torch.load(os.path.join("data", suffix, f"test_loader_{suffix}.pt"))

    # Determine the covariate dimension from the training loader.
    # For autoregressive DeepAR, each sample is a tuple of (target, covariates, mask).
    # We take the covariate tensor from the first sample.
    sample = next(iter(train_loader))
    # sample[1] is the covariate tensor with shape: [batch_size, T, covariate_dim]
    covariate_dim = sample[1].shape[-1]

    # Define the hyperparameter grid.
    hyperparameter_grid = {
        "covariate_size": [covariate_dim],
        "hidden_size": [20, 40, 60],
        "num_layers": [1, 2, 3],
        "embedding_dim": [32, 64],
        "learning_rate": [1e-3]
    }

    # Run grid search.
    best_config, best_loss, results = grid_search(
        hyperparameter_grid,
        train_loader,
        val_loader,
        device=device,
        num_epochs=50
    )

    # Print results.
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
