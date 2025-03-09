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
    'gmm': PriorWeightGMM,
    'tmm': PriorWeightTMM,
    'cauchy_mm': PriorWeightCauchyMM
}


def main(dist_name: str, dist_params: dict, hyperparam_path, spatial_arg='non_spatial', device='cuda'):
    spatial = True if spatial_arg == 'spatial' else False
    with open(hyperparam_path, 'r') as f:
        hyperparams = json.load(f)

    train_tensor = torch.load(
        f'data/{spatial_arg}/train_tensor_{spatial_arg}.pt')
    print('HERE:', train_tensor.shape)

    # prior_dist = PriorWeightGMM(
    #     proportions=torch.Tensor([dist_params['pi'], 1 - dist_params['pi']]),
    #     stds=torch.Tensor([dist_params['std1'], dist_params['std2']]), locs=[0, dist_params['mu2']])

    prior_dist = priors[dist_name](**dist_params)
    prior_dist.fit_params_to_data(data=train_tensor[:, 0])
    hyperparams['prior_dist'] = prior_dist

    # suffix = 'spatial' if spatial else 'non_spatial'

    save_suff = f"{dist_name}_{'_'.join([f'{k}_{v}' for k, v in dist_params.items(
    ) if v is not None and k != 'nus'])}"

    # Special handling for 'nus' parameter (for TMM)
    if 'nus' in dist_params and dist_params['nus'] is not None:
        # Extract the first value since all values are the same
        nu_val = dist_params['nus'][0].item()
        save_suff += f"_nu_{nu_val}"

    model = BSMDeTWrapper(**hyperparams)
    trained_model = bnn_single_model.main(
        terminal_args=False,
        hyperparam_path=hyperparam_path,
        train_epochs=1,
        device=device,
        suffix=spatial_arg,
        save_suff=save_suff,
        model=model
    )

    results = bnn_inference.main(
        model_path=f'modelsave/bmdet/bmdet_{spatial_arg}_{save_suff}.pt',
        test_loader_path=f'data/{spatial_arg}/test_loader_{spatial_arg}.pt',
        train_norm_path=f'data/{spatial_arg}/transforms_{spatial_arg}.pt',
        raw_csv_path='',
        num_samples=40,
        new_csv_savename=f'results/bmdet_{spatial_arg}_{save_suff}_stats.csv',
        test_start_date=datetime(2024, 9, 1),
        test_end_date=datetime(2025, 1, 6),
        device=device,
        plot_name=f'{spatial_arg}_{save_suff}',
        model=trained_model
    )
    try:
        os.makedirs('results', exist_ok=True)
        with open(f'results/metrics_{spatial_arg}_{save_suff}.json', 'w') as f:
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
    args = parser.parse_args()

    # dist_param_grid = {'pi': [0.25, 0.5, 0.75, 1]}
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

                    # For TMM, we need to create a tensor of nu values, all with the same value
                    if 'n' in tmm_params:
                        n = tmm_params['n']
                        # Create a tensor of nu values, all with the same value
                        # Use the device parameter if it exists in the params
                        device_param = tmm_params.get('device', 'cpu')
                        tmm_params['nus'] = torch.ones(
                            n, device=torch.device(device_param)) * nu_val

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
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print(f"Results for {dist_name}: {results}")
        else:
            # Standard handling for other distributions
            for params in param_combinations:
                print(
                    f"\nRunning experiment with distribution: {dist_name}, parameters: {params}")
                results = main(
                    dist_name=dist_name,
                    dist_params=params,
                    hyperparam_path=f'cfgs/bmdet/best_hyperparams_{args.spatial}.json',
                    spatial_arg=args.spatial,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print(f"Results for {dist_name}: {results}")
