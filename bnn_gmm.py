import json
import torch
import itertools
import os
from datetime import datetime
import argparse

import bnn_single_model
import bnn_inference

from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.distributions import PriorWeightGMM
from data_proc import StandardScaleNorm, MinMaxNorm


def main(dist_params: dict, hyperparam_path, spatial_arg='non_spatial', device='cuda'):
    # Convert spatial string arg to boolean
    spatial = True if spatial_arg == 'spatial' else False

    # Create GMM prior distribution
    prior_dist = PriorWeightGMM(
        proportions=torch.Tensor([dist_params['pi'], 1 - dist_params['pi']]),
        stds=torch.Tensor([dist_params['std1'], dist_params['std2']]), locs=[0, dist_params['mu2']])

    suffix = 'spatial' if spatial else 'non_spatial'

    # Load and update hyperparameters with prior distribution
    with open(hyperparam_path, 'r') as f:
        hyperparams = json.load(f)
    hyperparams['prior'] = prior_dist
    print(type(hyperparams['prior']), hyperparams['prior'].PI)

    # Create save suffix with distribution parameters
    save_suff = f"gmm_pi{dist_params['pi']}_std1{dist_params['std1']}_std2{dist_params['std2']}"
    model = BSMDeTWrapper(**hyperparams)
    print('HERE0:', model.prior.PI)
    # Train model
    trained_model = bnn_single_model.main(
        terminal_args=False,
        hyperparam_path=hyperparam_path,
        train_epochs=1,
        device=device,
        suffix=suffix,
        save_suff=save_suff,
        model=model
    )

    # Evaluate model
    results = bnn_inference.main(
        model_path=f'modelsave/bmdet/bmdet_{suffix}_{save_suff}.pt',
        test_loader_path=f'data/{suffix}/test_loader_{suffix}.pt',
        train_norm_path=f'data/{suffix}/transforms_{suffix}.pt',
        raw_csv_path='',
        num_samples=40,
        new_csv_savename=f'results/bmdet_{suffix}_{save_suff}_stats.csv',
        test_start_date=datetime(2024, 9, 1),
        test_end_date=datetime(2025, 1, 6),
        device=device,
        plot_name=f'{suffix}_{save_suff}',
        model=trained_model
    )
    try:
        os.makedirs('results', exist_ok=True)
        with open(f'results/metrics_{suffix}_{save_suff}.json', 'w') as f:
            # Convert numpy/torch values to native Python types for JSON serialization
            results_serializable = {k: float(v) for k, v in results.items()}
            json.dump(results_serializable, f, indent=4)
    except:
        print("COULD NOT SAVE EVAL RESULTS")

    # Clean up GPU memory
    if device != 'cpu':
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spatial', type=str, default='non_spatial',
                        choices=['spatial', 'non_spatial'],
                        help='Whether to use spatial features')
    args = parser.parse_args()

    dist_param_grid = {'pi': [0.5, 0.7, 1],
                       'std1': [1], 'std2': [0.3, 1.5, 2, 5], 'mu2': [-0.5, 0, 0.5]}

    param_combinations = [{'pi': pi, 'std1': s1, 'std2': s2, 'mu2': m2}
                          for pi in dist_param_grid['pi']
                          for s1 in dist_param_grid['std1']
                          for s2 in dist_param_grid['std2']
                          for m2 in dist_param_grid['mu2']]

    for params in param_combinations:
        print(f"\nRunning experiment with parameters: {params}")
        results = main(
            dist_params=params,
            hyperparam_path='modelsave/bmdet/best_hyperparams_non_spatial.json',
            spatial_arg=args.spatial,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Results: {results}")
