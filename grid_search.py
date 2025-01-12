import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools


def grid_search_torch_model(model_class: nn.Module, trainer_class, param_grid: dict, train_loader, test_loader, criterion, device='cpu'):
    param_combinations = list(itertools.product(*param_grid.values()))
    best_model = None
    best_params = None
    best_score = float('inf')

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        model = model_class(**param_dict).to(device)
        trainer = trainer_class(lr=param_dict.get(
            'lr', 0.001), epochs=param_dict.get('epochs', 10), model=model)

        trainer.train()

        val_loss = trainer.test(criterion=criterion)

        if val_loss < best_score:
            best_score = val_loss
            best_model = model
            best_params = param_dict
