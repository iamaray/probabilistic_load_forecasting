import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_proc import StandardScaleNorm, MinMaxNorm
from bayes_transformer.distributions import PriorWeightGMM
from bayes_transformer.model import BSMDeTWrapper
import bnn_inference
import bnn_single_model
import json
import torch
import itertools
from datetime import datetime
import argparse



def main(dist_params: dict, hyperparam_path, spatial_arg='non_spatial', device='cuda'):
    spatial = True if spatial_arg == 'spatial' else False

    prior_dist = PriorWeightGMM(
        proportions=torch.Tensor([dist_params['pi'], 1 - dist_params['pi']]),
        stds=torch.Tensor([dist_params['std1'], dist_params['std2']]), locs=[0, dist_params['mu2']])

    suffix = 'spatial' if spatial else 'non_spatial'

    with open(hyperparam_path, 'r') as f:
        hyperparams = json.load(f)
    hyperparams['prior'] = prior_dist
    print(type(hyperparams['prior']), hyperparams['prior'].PI)

    save_suff = f"gmm_pi{dist_params['pi']}"
    model = BSMDeTWrapper(**hyperparams)
    print('HERE0:', model.prior.PI)
    trained_model = bnn_single_model.main(
        terminal_args=False,
        hyperparam_path=hyperparam_path,
        train_epochs=1,
        device=device,
        suffix=suffix,
        save_suff=save_suff,
        model=model
    )

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
    args = parser.parse_args()

    dist_param_grid = {'pi': [0.25, 0.5, 0.75, 1]}

    param_combinations = [{'pi': pi}
                          for pi in dist_param_grid['pi']]

    for params in param_combinations:
        print(f"\nRunning experiment with parameters: {params}")
        results = main(
            dist_params=params,
            hyperparam_path='modelsave/bmdet/best_hyperparams_non_spatial.json',
            spatial_arg=args.spatial,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Results: {results}")
