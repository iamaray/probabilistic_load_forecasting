import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from bayes_transformer.model import BSMDeTWrapper
import json


def grid_search_torch_model(model_class: nn.Module, trainer_class, param_grid: dict, training_args: dict, train_loader, test_loader, criterion=None, device='cpu', savedir='BSMDeT_results'):
    param_combinations = list(itertools.product(*param_grid.values()))
    best_model = None
    best_params = None
    best_score = float('inf')

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        model = model_class(**param_dict).to(device)
        trainer = trainer_class(model_wrapper=model, train_loader=train_loader)

        trainer.train(**training_args)
        # model_class.test()
        val_loss = trainer.test(test_loader=test_loader)

        # val_loss = None

        # val_loss = trainer.test(criterion=criterion, test_loader=test_loader)
        # if isinstance(val_loss, list):
        #     val_loss = np.mean(val_loss)
        print(f'Computed val loss of {val_loss}, comparing with {best_score}.')
        if val_loss < best_score:
            best_score = val_loss
            best_model = model
            best_params = param_dict

    torch.save(best_model.state_dict(), f'{savedir}/best_model_params.pth')
    with open(f'{savedir}/best_hyperparams.json', 'w') as f:
        json.dump(best_params, f)
