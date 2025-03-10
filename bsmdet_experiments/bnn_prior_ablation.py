import argparse
from datetime import datetime
import itertools
import torch
import json
import bnn_single_model
import bnn_inference
from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.distributions import PriorWeightGMM, PriorWeightTMM, PriorWeightPhaseType, PriorWeightGaussian, PriorWeightStudentT, PriorWeightCauchyMM
from data_proc import StandardScaleNorm, MinMaxNorm, TransformSequence
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


priors = {
    # 'gmm': PriorWeightGMM,
    'tmm': PriorWeightTMM
    # 'cauchy_mm': PriorWeightCauchyMM
}


def main(dist_name: str, dist_params: dict, hyperparam_path, spatial_arg='non_spatial', device='cuda', experiment_config=None):
    """
    Main function to run experiments with different prior distributions.

    Args:
        dist_name: Name of the distribution to use
        dist_params: Parameters for the distribution
        hyperparam_path: Path to the hyperparameters JSON file
        spatial_arg: Whether to use spatial features
        device: Device to use for training
        experiment_config: Configuration for the experiment
    """
    # Use default values if experiment_config is not provided
    if experiment_config is None:
        experiment_config = {
            "training": {"train_epochs": 1, "save_model": True},
            "inference": {
                "num_samples": 40,
                "test_start_date": "2024-09-01",
                "test_end_date": "2025-01-06"
            },
            "paths": {
                "model_save_dir": "modelsave/bmdet",
                "results_dir": "results",
                "data_dir": "data"
            },
            "device": device
        }

    # Override device if specified in experiment_config
    device = experiment_config.get("device", device)

    # Extract paths from config
    paths = experiment_config.get("paths", {})
    model_save_dir = paths.get("model_save_dir", "modelsave/bmdet")
    results_dir = paths.get("results_dir", "results")
    data_dir = paths.get("data_dir", "data")

    # Extract training parameters
    training_config = experiment_config.get("training", {})
    train_epochs = training_config.get("train_epochs", 1)
    save_model = training_config.get("save_model", True)

    # Extract inference parameters
    inference_config = experiment_config.get("inference", {})
    num_samples = inference_config.get("num_samples", 40)
    test_start_date_str = inference_config.get("test_start_date", "2024-09-01")
    test_end_date_str = inference_config.get("test_end_date", "2025-01-06")

    # Parse dates
    test_start_date = datetime.fromisoformat(test_start_date_str)
    test_end_date = datetime.fromisoformat(test_end_date_str)

    spatial = True if spatial_arg == 'spatial' else False
    with open(hyperparam_path, 'r') as f:
        hyperparams = json.load(f)

    train_tensor = torch.load(
        f'{data_dir}/{spatial_arg}/train_tensor_{spatial_arg}.pt')

    # Move train_tensor to the specified device
    train_tensor = train_tensor.to(device)

    # Ensure device is passed to the prior distribution
    if 'device' not in dist_params:
        dist_params['device'] = device

    prior_dist = priors[dist_name](**dist_params)
    prior_dist.fit_params_to_data(data=train_tensor[:, 0])
    hyperparams['prior_dist'] = prior_dist

    save_suff = f"{dist_name}_{'_'.join([f'{k}_{v}' for k, v in dist_params.items(
    ) if v is not None and k != 'nus'])}"

    # Special handling for 'nus' parameter (for TMM)
    if 'nus' in dist_params and dist_params['nus'] is not None:
        # Extract the first value since all values are the same
        nu_val = dist_params['nus'][0].item()
        save_suff += f"_nu_{nu_val}"

    model = BSMDeTWrapper(**hyperparams)

    # Create model save path
    model_path = f'{model_save_dir}/bmdet_{spatial_arg}_{save_suff}.pt'

    trained_model = bnn_single_model.main(
        terminal_args=False,
        hyperparam_path=hyperparam_path,
        train_epochs=train_epochs,
        device=device,
        suffix=spatial_arg,
        save_suff=save_suff,
        model=model
    )

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Define paths for inference
    results_csv_path = f'{results_dir}/bmdet_{spatial_arg}_{save_suff}_stats.csv'

    results = bnn_inference.main(
        model_path=model_path,
        test_loader_path=f'{data_dir}/{spatial_arg}/test_loader_{spatial_arg}.pt',
        train_norm_path=f'{data_dir}/{spatial_arg}/transforms_{spatial_arg}.pt',
        raw_csv_path='',
        num_samples=num_samples,
        new_csv_savename=results_csv_path,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        device=device,
        plot_name=f'{spatial_arg}_{save_suff}',
        model=trained_model
    )

    try:
        metrics_path = f'{results_dir}/metrics_{spatial_arg}_{save_suff}.json'
        with open(metrics_path, 'w') as f:
            results_serializable = {k: float(v) for k, v in results.items()}
            json.dump(results_serializable, f, indent=4)
    except:
        print("COULD NOT SAVE EVAL RESULTS")

    if device != 'cpu':
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial', type=str, default='non_spatial',
                        choices=['spatial', 'non_spatial'],
                        help='Whether to use spatial features')
    parser.add_argument('--prior_param_grid_path', type=str,
                        help='Path to the prior parameter grid')
    parser.add_argument('--experiment_config_path', type=str, default=None,
                        help='Path to the experiment configuration JSON file')
    args = parser.parse_args()

    # Load experiment configuration if provided
    experiment_config = None
    if args.experiment_config_path:
        with open(args.experiment_config_path, 'r') as f:
            experiment_config = json.load(f)
            print(
                f"Loaded experiment configuration from {args.experiment_config_path}")

    # Load prior parameter grid
    with open(args.prior_param_grid_path, 'r') as f:
        dist_param_grid = json.load(f)

    # Create a grid of parameter combinations where each dictionary has a single value for each parameter
    param_combinations = []

    # Get all parameter names and their possible values
    param_names = list(dist_param_grid.keys())
    param_values = [dist_param_grid[name]
                    for name in param_names if dist_param_grid[name] is not None]

    # Generate all combinations using nested loops
    def generate_combinations(current_combo, param_idx):
        if param_idx >= len(param_names):
            param_combinations.append(current_combo.copy())
            return

        if dist_param_grid[param_names[param_idx]] is None:
            generate_combinations(current_combo, param_idx + 1)
        else:
            for val in dist_param_grid[param_names[param_idx]]:
                current_combo[param_names[param_idx]] = val
                generate_combinations(current_combo, param_idx + 1)

    generate_combinations({}, 0)
    print(param_combinations)

    # Get device from experiment config if available
    device_to_use = 'cuda' if torch.cuda.is_available() else 'cpu'
    if experiment_config and "device" in experiment_config:
        device_to_use = experiment_config["device"]

    # Loop through each distribution type
    for dist_name in priors.keys():
        print(f"\n=== Testing distribution: {dist_name} ===")

        # Special handling for Student's t mixture model
        if dist_name == 'tmm' and 'nu' in dist_param_grid:
            tmm_param_combinations = []

            # For each combination of parameters
            for params in param_combinations:
                # For each nu value
                for nu_val in dist_param_grid['nu']:
                    # Create a new parameter set with the nu value
                    tmm_params = params.copy()

                    # Remove the 'nu' parameter since we'll use 'nus' instead
                    if 'nu' in tmm_params:
                        del tmm_params['nu']

                    # Override the device parameter to ensure consistency
                    tmm_params['device'] = device_to_use

                    # For TMM, we need to create a tensor of nu values, all with the same value
                    if 'n' in tmm_params:
                        n = tmm_params['n']
                        # Create a tensor of nu values, all with the same value on the correct device
                        tmm_params['nus'] = torch.ones(
                            n, device=torch.device(device_to_use)) * nu_val

                    tmm_param_combinations.append(tmm_params)

            # Use the TMM-specific parameter combinations
            for params in tmm_param_combinations:
                print(
                    f"\nRunning experiment with distribution: {dist_name}, parameters: {params}")
                results = main(
                    dist_name=dist_name,
                    dist_params=params,
                    hyperparam_path=f'cfgs/bmdet/best_hyperparams_{args.spatial}.json',
                    spatial_arg=args.spatial,
                    device=device_to_use,
                    experiment_config=experiment_config
                )
                print(f"Results for {dist_name}: {results}")
        else:
            # Standard handling for other distributions
            for params in param_combinations:
                # Create a filtered copy of params without irrelevant parameters
                filtered_params = params.copy()

                # Remove parameters that aren't relevant to this distribution type
                if dist_name != 'tmm' and 'nu' in filtered_params:
                    del filtered_params['nu']

                # Override the device parameter to ensure consistency
                filtered_params['device'] = device_to_use

                print(
                    f"\nRunning experiment with distribution: {dist_name}, parameters: {filtered_params}")
                results = main(
                    dist_name=dist_name,
                    dist_params=filtered_params,
                    hyperparam_path=f'cfgs/bmdet/best_hyperparams_{args.spatial}.json',
                    spatial_arg=args.spatial,
                    device=device_to_use,
                    experiment_config=experiment_config
                )
                print(f"Results for {dist_name}: {results}")
