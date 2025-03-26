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
from .model import DeepAR, neg_gaussian_log_likelihood

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
    def __init__(self, model, optimizer, train_loader, val_loader=None, device='cpu', train_norm=None):
        """
        Trainer for the DeepAR model.

        Args:
          model: An instance of the DeepAR model.
          optimizer: A PyTorch optimizer (e.g., Adam) for updating model parameters.
          train_loader: DataLoader for training data. Each batch contains target, covariates, mask.
          val_loader: (Optional) DataLoader for validation data.
          device: Device to run the model on ('cpu' or 'cuda').
          train_norm: Data transformation for normalization.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.train_norm = train_norm

    def train_epoch(self):
        """Run one epoch of training."""
        self.optimizer.zero_grad()

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # From DataLoader we get:
            # batch[0]: target sequence - Shape: (batch_size, seq_len)
            # batch[1]: covariates - Shape: (batch_size, seq_len, covariate_size)
            # batch[2]: mask - Shape: (batch_size, seq_len) - boolean mask

            # Get batch components and move to device
            target = batch[0].to(self.device)
            covariates = batch[1].to(self.device)
            # This indicates which parts of the sequence are observed
            mask = batch[2].to(self.device)

            batch_size = target.shape[0]
            seq_len = target.shape[1]

            # Create default time series index (zeros) if your model needs it
            idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            # Initialize hidden and cell states
            h = self.model.init_hidden(batch_size)
            c = self.model.init_cell(batch_size)

            # Full sequence loss
            loss = torch.zeros(1, device=self.device)

            # Process the sequence step by step as required by the model
            # This follows the model's expected interface
            for t in range(seq_len):
                # Skip the loss calculation at the prediction window
                if not mask[:, t].any():
                    continue

                # Handle missing values by replacing them with previous predictions
                zero_index = (target[:, t] == 0)
                if t > 0 and torch.sum(zero_index) > 0:
                    target[:, t][zero_index] = mu[zero_index]

                # Process one step through the model
                # Note: need to adjust dimensions to match model expectations
                # Model expects target (1, batch) and covariates (1, batch, cov_size)
                current_target = target[:, t].unsqueeze(0)
                current_covariates = covariates[:, t, :].unsqueeze(0)

                # Forward pass through the model
                mu, sigma, h, c = self.model(
                    target=current_target,
                    covariates=current_covariates,
                    idx=idx.unsqueeze(0),
                    h=h,
                    c=c
                )

                # Calculate loss only for observed time steps (where mask is True)
                step_mask = mask[:, t]
                if step_mask.any():
                    # Use negative log likelihood for Gaussian distribution
                    step_loss = neg_gaussian_log_likelihood(
                        mu, sigma, target[:, t])
                    loss += step_loss.sum()

            loss.backward()
            self.optimizer.step()

            # Normalize the loss by the number of observed time steps
            observed_points = mask.sum()
            if observed_points > 0:
                loss = loss / observed_points

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self):
        """Run one epoch of validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Similar process as training but without backpropagation
                target = batch[0].to(self.device)
                covariates = batch[1].to(self.device)
                mask = batch[2].to(self.device)

                # print(target.shape, covariates.shape)

                transformed = torch.cat(
                    [target.unsqueeze(-1), covariates], dim=-1)
                transformed = self.train_norm.transform(
                    transformed)

                target = transformed[:, :, 0]
                covariates = transformed[:, :, 1:]

                # print(target.shape, covariates.shape)

                batch_size = target.shape[0]
                seq_len = target.shape[1]

                idx = torch.zeros(
                    batch_size, dtype=torch.long, device=self.device)
                h = self.model.init_hidden(batch_size)
                c = self.model.init_cell(batch_size)

                loss = torch.zeros(1, device=self.device)

                for t in range(seq_len):
                    if not mask[:, t].any():
                        continue

                    zero_index = (target[:, t] == 0)
                    if t > 0 and torch.sum(zero_index) > 0:
                        target[:, t][zero_index] = mu[zero_index]

                    current_target = target[:, t].unsqueeze(0)
                    current_covariates = covariates[:, t, :].unsqueeze(0)

                    mu, sigma, h, c = self.model(
                        target=current_target,
                        covariates=current_covariates,
                        idx=idx.unsqueeze(0),
                        h=h,
                        c=c
                    )

                    step_mask = mask[:, t]
                    if step_mask.any():
                        step_loss = neg_gaussian_log_likelihood(
                            mu, sigma, target[:, t])
                        loss += step_loss.sum()

                observed_points = mask.sum()
                if observed_points > 0:
                    loss = loss / observed_points

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self, num_epochs):
        """Run the training loop for a given number of epochs."""
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            if self.val_loader is not None:
                val_loss = self.validate_epoch()
                val_losses.append(val_loss)
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        return train_losses, val_losses

    def save_model(self, savepath='modelsave', savename='deepar_model.pt'):
        """Save the model to disk."""
        os.makedirs(savepath, exist_ok=True)
        model_path = os.path.join(savepath, savename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


def grid_search(hyperparameter_grid, train_loader, val_loader, device='cpu', data_norm=None, num_epochs=10, savename='deepar_best_model_spatial'):
    """
    Performs a grid search over the hyperparameter grid for the DeepAR model.

    Args:
        hyperparameter_grid (dict): A dictionary where each key is a hyperparameter name and each value 
                                    is a list of possible values.
        train_loader: PyTorch DataLoader for training data.
        val_loader: PyTorch DataLoader for validation data.
        device (str): 'cpu' or 'cuda'.
        data_norm: Data normalization transforms.
        num_epochs (int): Number of epochs to train each model.
        savename (str): Base name to save the best model.

    Returns:
        best_model, best_config, best_loss, results
    """
    best_config = None
    best_loss = float('inf')
    results = []

    keys = list(hyperparameter_grid.keys())
    values = [hyperparameter_grid[key] for key in keys]

    # Check shape of the data
    sample_batch = next(iter(train_loader))
    target_shape = sample_batch[0].shape
    covariate_shape = sample_batch[1].shape

    print(f"Sample target shape: {target_shape}")
    print(f"Sample covariates shape: {covariate_shape}")

    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        print(f"Testing configuration: {config}")

        # Create model with current config
        model = DeepAR(
            num_class=config.get("num_class", 1),
            embedding_dim=config.get("embedding_dim", 32),
            covariate_size=config["covariate_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            predict_steps=config.get("predict_steps", 24),
            predict_start=config.get("predict_start", 168)
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        trainer = DeepARTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            train_norm=data_norm
        )

        # Train for specified number of epochs
        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch()
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # Evaluate on validation set
        val_loss = trainer.validate_epoch()
        print(f"Config {config} achieved validation loss: {val_loss:.4f}")

        results.append((config, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config

    print(
        f"\nBest configuration: {best_config} with validation loss: {best_loss:.4f}")

    # Train best model for additional epochs
    extra_epochs = 50
    print(f"\nTraining best model for {extra_epochs} additional epochs...")

    best_model = DeepAR(
        num_class=best_config.get("num_class", 1),
        embedding_dim=best_config.get("embedding_dim", 32),
        covariate_size=best_config["covariate_size"],
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"],
        predict_steps=best_config.get("predict_steps", 24),
        predict_start=best_config.get("predict_start", 168)
    ).to(device)

    best_optimizer = optim.Adam(
        best_model.parameters(), lr=best_config["learning_rate"])

    best_trainer = DeepARTrainer(
        model=best_model,
        optimizer=best_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        train_norm=data_norm
    )

    for epoch in range(num_epochs + extra_epochs):
        train_loss = best_trainer.train_epoch()
        val_loss = best_trainer.validate_epoch()
        print(
            f"Training best model -- Epoch {epoch+1}/{num_epochs + extra_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Create modelsave directory if it doesn't exist
    os.makedirs('modelsave', exist_ok=True)

    # Save the model and hyperparameters
    model_path = os.path.join('modelsave', f'{savename}.pt')
    config_path = os.path.join('modelsave', f'{savename}_cfg.json')

    torch.save(best_model.state_dict(), model_path)
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=4)

    print(f"\nBest model saved to: {model_path}")
    print(f"Hyperparameters saved to: {config_path}")

    return best_model, best_config, best_loss, results
