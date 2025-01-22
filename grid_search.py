import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from bayes_transformer.model import BSMDeTWrapper
import json


def grid_search_torch_model(
        model_class: nn.Module,
        trainer_class,
        param_grid: dict,
        training_args: dict,
        train_loader,
        test_loader,
        criterion=None,
        device='cpu',
        savedir='modelsave/bmdet/',
        savename='bmdet_best_model.pt',
        train_norm=None,
        test_norm=None):

    param_combinations = list(itertools.product(*param_grid.values()))
    best_model = None
    best_params = None
    best_score = float('inf')

    best_trainer = None
    for params in param_combinations:

        print(f"Evaluating params: {params}")

        param_dict = dict(zip(param_grid.keys(), params))
        # model = model_class(**param_dict).to(device)
        model = model_class(**param_dict)
        trainer = trainer_class(
            model_wrapper=model, train_loader=train_loader, train_norm=train_norm, test_norm=test_norm)

        trainer.train(**training_args)
        val_loss = trainer.test(test_loader=test_loader)

        print(f'Computed val loss of {val_loss}, comparing with {best_score}.')
        if val_loss < best_score:
            best_score = val_loss
            best_model = model
            best_params = param_dict
            best_trainer = trainer

    # torch.save(best_model.state_dict(), f'{savedir}/best_model_params.pth')
    if best_trainer is not None:
        best_trainer.save_model(savepath=savedir, savename=savename)
    else:
        print('Best model NOT saved :(')

    # if isinstance(best_model, BSMDeTWrapper):
    #     torch.save(model.model)

    with open(f'{savedir}/best_hyperparams.json', 'w') as f:
        json.dump(best_params, f)
