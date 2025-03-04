# from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
import argparse
import logging
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import itertools
import utils
from .model import DeepAR

from metrics import compute_metrics
# from evaluate import evaluate
# from dataloader import *

# import matplotlib
# matplotlib.use('Agg')

# logger = logging.getLogger('DeepAR.Train')

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='elect', help='Name of the dataset')
# parser.add_argument('--data-folder', default='data',
#                     help='Parent dir of the dataset')
# parser.add_argument('--model-name', default='base_model',
#                     help='Directory containing params.json')
# parser.add_argument('--relative-metrics', action='store_true',
#                     help='Whether to normalize the metrics by label scales')
# parser.add_argument('--sampling', action='store_true',
#                     help='Whether to sample during evaluation')
# parser.add_argument('--save-best', action='store_true',
#                     help='Whether to save best ND to param_search.txt')
# parser.add_argument('--restore-file', default=None,
#                     # 'best' or 'epoch_#'
#                     help='Optional, name of the file in --model_dir containing weights to reload before training')
# parser.add_argument('--save-file', action='store_true',
#                     help='Whether to save during evaluation')


# class DeepARTrainer:
#     def __init__(self, model: nn.Module, train_loader: DataLoader,
#                  test_loader: DataLoader, device: str = 'cuda'):
#         self.model = model
#         self.optimizer = optim.Adam(params=model.parameters(), lr=0.001)
#         self.loss_fn = loss_fn
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.device = device
#         self.train_window = 168

#     def train(self, num_epochs: int) -> list:
#         '''Train the model for multiple epochs.
#         Args:
#             num_epochs: (int) number of epochs to train for
#         Returns:
#             list: array of losses for each epoch
#         '''
#         losses = []
#         for epoch in range(num_epochs):
#             print(f'Epoch {epoch+1}/{num_epochs}')
#             epoch_loss = self._train_epoch(epoch)
#             losses.append(epoch_loss)
#         return losses

#     def _train_epoch(self, epoch: int) -> float:
#         '''Train the model on one epoch by batches.
#         Args:
#             epoch: (int) the current training epoch
#         Returns:
#             float: array of losses for each batch
#         '''
#         self.model.train()
#         loss_epoch = np.zeros(len(self.train_loader))
#         # Train_loader:
#         # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
#         # idx ([batch_size]): one integer denoting the time series id;
#         # labels_batch ([batch_size, train_window]): z_{1:T}.

#         for i, (train_batch, labels_batch) in enumerate(tqdm(self.train_loader)):
#             self.optimizer.zero_grad()
#             batch_size = train_batch.shape[0]

#             train_batch = train_batch.permute(1, 0, 2).to(
#                 torch.float32).to(self.device)
#             labels_batch = labels_batch.squeeze(1)
#             print(labels_batch.shape)
#             labels_batch = labels_batch.permute(1, 0).to(
#                 torch.float32).to(self.device)
#             idx = torch.zeros((batch_size)).int()
#             idx = idx.unsqueeze(0).to(self.device)

#             loss = torch.zeros(1, device=self.device)
#             hidden = self.model.init_hidden(batch_size)
#             cell = self.model.init_cell(batch_size)

#             for t in range(self.train_window):
#                 # if z_t is missing, replace it by output mu from the last time step
#                 zero_index = (train_batch[t, :, 0] == 0)
#                 if t > 0 and torch.sum(zero_index) > 0:
#                     train_batch[t, zero_index, 0] = mu[zero_index]
#                 mu, sigma, hidden, cell = self.model(
#                     x=train_batch[t].unsqueeze_(0).clone(), idx=idx, hidden=hidden, cell=cell)
#                 loss += self.loss_fn(mu, sigma, labels_batch[t])

#             loss.backward()
#             self.optimizer.step()
#             # loss per timestep
#             loss = loss.item() / self.train_window
#             loss_epoch[i] = loss
#             if i % 1000 == 0:
#                 test_metrics = evaluate(
#                     self.model, self.loss_fn, self.test_loader, self.params, epoch, sample=args.sampling)
#                 self.model.train()
#                 logger.info(f'train_loss: {loss}')
#             if i == 0:
#                 logger.info(f'train_loss: {loss}')
#         return loss_epoch


# def grid_search_torch_model(
#         model_class: nn.Module,
#         trainer_class,
#         param_grid: dict,
#         training_args: dict,
#         train_loader,
#         test_loader,
#         criterion=None,
#         device='cpu',
#         savedir='modelsave/bmdet/',
#         savename='bmdet_best_model.pt',
#         train_norm=None,
#         test_norm=None):

#     param_combinations = list(itertools.product(*param_grid.values()))
#     best_model = None
#     best_params = None
#     best_acr_diff = float('inf')
#     best_trainer = None

#     for params in param_combinations:
#         print(len(param_combinations))
#         print(f"Evaluating params: {params}")

#         param_dict = dict(zip(param_grid.keys(), params))
#         # model = model_class(**param_dict).to(device)
#         model = model_class(**param_dict)
#         trainer = trainer_class(
#             model_wrapper=model, train_loader=train_loader, train_norm=train_norm)

#         trainer.train(**training_args)
#         # val_loss = trainer.test(test_loader=test_loader)

#         outs = []
#         # for (x, y) in test_loader:
#         #     out = model.test(in_test=x.to(device),
#         #                      samples=20, scaler=train_norm)
#         #     outs.append(out)
#         #     y = y.transpose(1, 2)
#         #     metrics.append(compute_metrics(out, y))
#         metrics = model.test(test_loader=test_loader, sampling=True)

#         # closeness to 0.8
#         acr = metrics['ACR']
#         acr_diff = np.abs(0.8 - acr)

#         print(
#             f'Computed val loss of {acr_diff}, comparing with {best_acr_diff}.')

#         if acr_diff < best_acr_diff:
#             best_acr_diff = acr_diff
#             best_model = model
#             best_params = param_dict
#             best_trainer = trainer

#     # torch.save(best_model.state_dict(), f'{savedir}/best_model_params.pth')
#     if best_trainer is not None:
#         best_trainer.save_model(savepath=savedir, savename=savename)
#     else:
#         print('Best model NOT saved :(')

#     # if isinstance(best_model, BSMDeTWrapper):
#     #     torch.save(model.model)

#     with open(f'{savedir}/best_hyperparams.json', 'w') as f:
#         json.dump(best_params, f)


import torch
from itertools import product


class DeepARTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader=None, device='cpu'):
        """
        Args:
          model: An instance of the DeepAR model.
          optimizer: A PyTorch optimizer (e.g., Adam) for updating model parameters.
          train_loader: DataLoader for training data. Each batch is expected to be a dict with keys:
                        'target' (shape: [batch_size, seq_len]),
                        'covariates' (shape: [batch_size, seq_len, covariate_size]),
                        'mask' (shape: [batch_size, seq_len]) where True indicates observed data.
          val_loader: (Optional) DataLoader for validation data (same expected format as train_loader).
          device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_epoch(self):
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:

            # Shape: (batch_size, seq_len)
            targets = batch[0].to(self.device)
            # Shape: (batch_size, seq_len, covariate_size)
            covariates = batch[1].to(self.device)
            # Shape: (batch_size, seq_len)
            mask = batch[2].to(self.device)

            self.optimizer.zero_grad()
            loss, _ = self.model(targets, covariates, mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate_epoch(self, data_norm):
        """Run one epoch of validation."""
        self.model.eval()
        total_loss = 0.0
        data_norm.set_device(device=self.device)
        with torch.no_grad():
            for batch in self.val_loader:
                targets = data_norm.transform(
                    transform_col=0, x=batch[0].unsqueeze(-1).to(self.device)).squeeze()
                print('here', targets.shape)
                covariates = data_norm.transform(batch[1].to(self.device))
                mask = batch[2].to(self.device)

                loss, _ = self.model(targets, covariates, mask)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, num_epochs):
        """Run the training loop for a given number of epochs."""
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            if self.val_loader is not None:
                val_loss = self.validate_epoch()
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")


def grid_search(hyperparameter_grid, train_loader, val_loader, device='cpu', data_norm=None, num_epochs=10, savename='deepar_best_model_spatial'):
    """
    Performs a grid search over the hyperparameter grid for the DeepAR model.

    Args:
        hyperparameter_grid (dict): A dictionary where each key is a hyperparameter name and each value 
                                    is a list of possible values. Expected keys include:
                                    - "covariate_size": int, dimensionality of covariates.
                                    - "hidden_size": int, number of LSTM hidden units.
                                    - "num_layers": int, number of LSTM layers.
                                    - "learning_rate": float, learning rate for the optimizer.
        train_loader: PyTorch DataLoader for training data.
        val_loader: PyTorch DataLoader for validation data.
        device (str): 'cpu' or 'cuda'.
        num_epochs (int): Number of epochs to train each model.

    Returns:
        best_config (dict): The configuration with the lowest validation loss.
        best_loss (float): The lowest achieved validation loss.
        results (list): A list of tuples (config, validation_loss) for each grid combination.
    """
    best_config = None
    best_loss = float('inf')
    results = []

    keys = list(hyperparameter_grid.keys())
    values = [hyperparameter_grid[key] for key in keys]

    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        print(f"Testing configuration: {config}")

        model = DeepAR(covariate_size=config["covariate_size"],
                       hidden_size=config["hidden_size"],
                       num_layers=config["num_layers"])
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        trainer = DeepARTrainer(
            model, optimizer, train_loader, val_loader, device)

        for epoch in range(num_epochs):
            l = trainer.train_epoch()
            print(f"Epoch {epoch+1}/{num_epochs}: {l}")

        val_loss = trainer.validate_epoch(data_norm=data_norm)
        print(f"Config {config} achieved validation loss: {val_loss:.4f}")

        results.append((config, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config

    print(
        f"\nBest configuration: {best_config} with validation loss: {best_loss:.4f}")

    # Train best model for additional 20 epochs
    print("\nTraining best model for 20 additional epochs...")
    best_model = DeepAR(covariate_size=best_config["covariate_size"],
                        hidden_size=best_config["hidden_size"],
                        num_layers=best_config["num_layers"],
                        embedding_dim=best_config['embedding_dim'])
    best_model = best_model.to(device)
    best_optimizer = optim.Adam(
        best_model.parameters(), lr=best_config["learning_rate"])
    best_trainer = DeepARTrainer(
        best_model, best_optimizer, train_loader, val_loader, device)

    for epoch in range(num_epochs + 20):
        train_loss = best_trainer.train_epoch()
        val_loss = best_trainer.validate_epoch(data_norm=data_norm)
        print(
            f"Training best model -- Epoch {epoch+1}/{num_epochs + 20}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Create modelsave directory if it doesn't exist
    os.makedirs('modelsave', exist_ok=True)

    # Save the model and hyperparameters
    timestamp = utils.get_timestamp()
    model_path = os.path.join('modelsave', f'{savename}.pt')
    config_path = os.path.join(
        'modelsave', f'{savename}_cfg.json')

    torch.save(best_model.state_dict(), model_path)
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=4)

    print(f"\nBest model saved to: {model_path}")
    print(f"Hyperparameters saved to: {config_path}")

    return best_config, best_loss, results
