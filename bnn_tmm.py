import json
import torch
import itertools
import os
from datetime import datetime
import argparse

import bnn_single_model
import bnn_inference

from bayes_transformer.model import BSMDeTWrapper
from bayes_transformer.distributions import PriorWeightTMM
from data_proc import StandardScaleNorm, MinMaxNorm


def main(dist_params: dict, hyperparam_path, spatial_arg='non_spatial', device='cuda'):
    # Convert spatial string arg to boolean
    spatial = True if spatial_arg == 'spatial' else False

    # Create TMM prior distribution
    prior_dist = PriorWeightTMM(
        proportions=torch.Tensor([dist_params['pi'], 1 - dist_params['pi']]),
        nus=torch.Tensor([dist_params['nu1'], dist_params['nu2']]),
        sigmas=torch.Tensor([dist_params['sigma1'], dist_params['sigma2']]),
        mus=torch.Tensor([0, dist_params['mu2']])
    )

    # suffix = 'spatial' if spatial else 'non_spatial'

    try:
        # Load and update hyperparameters with prior distribution
        with open(hyperparam_path, 'r') as f:
            hyperparams = json.load(f)
        hyperparams['prior'] = prior_dist
        print(type(hyperparams['prior']), hyperparams['prior'].PI)
    except:
        print("COULD NOT SAVE EVAL REULTS")

    # Create save suffix with distribution parameters
    save_suff = f"tmm_pi{dist_params['pi']}_nu1{dist_params['nu1']}_nu2{dist_params['nu2']}_sigma1{dist_params['sigma1']}_sigma2{dist_params['sigma2']}_mu2{dist_params['mu2']}"
    model = BSMDeTWrapper(**hyperparams)
    print('HERE0:', model.prior.PI)
    # Train model
    trained_model = bnn_single_model.main(
        terminal_args=False,
        hyperparam_path=hyperparam_path,
        train_epochs=1,
        device=device,
        suffix=spatial_arg,
        save_suff=save_suff,
        model=model
    )

    # Evaluate model
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

    # Save evaluation results
    os.makedirs('results', exist_ok=True)
    with open(f'results/metrics_{spatial_arg}_{save_suff}.json', 'w') as f:
        # Convert numpy/torch values to native Python types for JSON serialization
        results_serializable = {k: float(v) for k, v in results.items()}
        json.dump(results_serializable, f, indent=4)

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
                       'nu1': [3, 5], 'nu2': [3, 5],
                       'sigma1': [1], 'sigma2': [0.3, 1.5, 2],
                       'mu2': [-0.5, 0, 0.5]}

    param_combinations = [{'pi': pi, 'nu1': n1, 'nu2': n2, 'sigma1': s1, 'sigma2': s2, 'mu2': m2}
                          for pi in dist_param_grid['pi']
                          for n1 in dist_param_grid['nu1']
                          for n2 in dist_param_grid['nu2']
                          for s1 in dist_param_grid['sigma1']
                          for s2 in dist_param_grid['sigma2']
                          for m2 in dist_param_grid['mu2']]

    for params in param_combinations:
        print(f"\nRunning experiment with parameters: {params}")
        results = main(
            dist_params=params,
            hyperparam_path=f'modelsave/bmdet/best_hyperparams_{args.spatial}.json',
            spatial_arg=args.spatial,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Results: {results}")
