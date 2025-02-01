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

import utils
from .model import Net, loss_fn

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


class DeepARTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, device: str = 'cuda'):
        self.model = model
        self.optimizer = optim.Adam(params=model.parameters(), lr=0.001)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.train_window = 168

    def train(self, num_epochs: int) -> list:
        '''Train the model for multiple epochs.
        Args:
            num_epochs: (int) number of epochs to train for
        Returns:
            list: array of losses for each epoch
        '''
        losses = []
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            epoch_loss = self._train_epoch(epoch)
            losses.append(epoch_loss)
        return losses

    def _train_epoch(self, epoch: int) -> float:
        '''Train the model on one epoch by batches.
        Args:
            epoch: (int) the current training epoch
        Returns:
            float: array of losses for each batch
        '''
        self.model.train()
        loss_epoch = np.zeros(len(self.train_loader))
        # Train_loader:
        # train_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
        # idx ([batch_size]): one integer denoting the time series id;
        # labels_batch ([batch_size, train_window]): z_{1:T}.

        for i, (train_batch, idx, labels_batch) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            batch_size = train_batch.shape[0]

            train_batch = train_batch.permute(1, 0, 2).to(
                torch.float32).to(self.device)  # not scaled
            labels_batch = labels_batch.permute(1, 0).to(
                torch.float32).to(self.device)  # not scaled
            idx = idx.unsqueeze(0).to(self.device)

            loss = torch.zeros(1, device=self.device)
            hidden = self.model.init_hidden(batch_size)
            cell = self.model.init_cell(batch_size)

            for t in range(self.train_window):
                # if z_t is missing, replace it by output mu from the last time step
                zero_index = (train_batch[t, :, 0] == 0)
                if t > 0 and torch.sum(zero_index) > 0:
                    train_batch[t, zero_index, 0] = mu[zero_index]
                mu, sigma, hidden, cell = self.model(
                    train_batch[t].unsqueeze_(0).clone(), idx, hidden, cell)
                loss += self.loss_fn(mu, sigma, labels_batch[t])

            loss.backward()
            self.optimizer.step()
            # loss per timestep
            loss = loss.item() / self.train_window
            loss_epoch[i] = loss
            if i % 1000 == 0:
                test_metrics = evaluate(
                    self.model, self.loss_fn, self.test_loader, self.params, epoch, sample=args.sampling)
                self.model.train()
                logger.info(f'train_loss: {loss}')
            if i == 0:
                logger.info(f'train_loss: {loss}')
        return loss_epoch


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
    best_acr_diff = float('inf')
    best_trainer = None

    for params in param_combinations:
        print(len(param_combinations))
        print(f"Evaluating params: {params}")

        param_dict = dict(zip(param_grid.keys(), params))
        # model = model_class(**param_dict).to(device)
        model = model_class(**param_dict)
        trainer = trainer_class(
            model_wrapper=model, train_loader=train_loader, train_norm=train_norm)

        trainer.train(**training_args)
        # val_loss = trainer.test(test_loader=test_loader)

        outs = []
        # for (x, y) in test_loader:
        #     out = model.test(in_test=x.to(device),
        #                      samples=20, scaler=train_norm)
        #     outs.append(out)
        #     y = y.transpose(1, 2)
        #     metrics.append(compute_metrics(out, y))
        metrics = model.test(test_loader=test_loader, sampling=True)

        # closeness to 0.8
        acr = metrics['ACR']
        acr_diff = np.abs(0.8 - acr)

        print(f'Computed val loss of {
              acr_diff}, comparing with {best_acr_diff}.')

        if acr_diff < best_acr_diff:
            best_acr_diff = acr_diff
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
