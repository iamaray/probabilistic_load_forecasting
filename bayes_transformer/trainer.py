import numpy as np
import torch
import torch.nn as nn
import datetime
import os
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor

from .model import BSMDeTWrapper
from .utils import loadDataforTrainVal  # if used
from preprocessing import MinMaxNorm, StandardScaleNorm
from metrics import compute_metrics

# ---------------------------
# Trainer Class
# ---------------------------


class BayesTrainer:
    def __init__(self, model_wrapper: BSMDeTWrapper = None, train_loader=None,
                 input_size=24*3, output_size=1, batch_size=64, epochs=100,
                 num_targets=1, num_aux_feats=0, window_len=168, ahead=1,
                 train_norm=None, modelsave=False, savename='bmdet', device=None):
        self.modelsave = modelsave
        self.savename = savename
        self.device = device if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(
            f"using GPU {self.device}" if torch.cuda.is_available() else "using CPU")

        self.train_loader = train_loader
        self.Nbatch = len(self.train_loader)

        # Model setup
        self.net = model_wrapper.to(self.device)
        total_params = np.sum(p.numel() for p in self.net.parameters())
        print('    Total params: %.2fM' % (total_params / 1e6))

        # Setup for saving model and logging metrics
        self.savepath = 'modelsave/bmdet/'
        os.makedirs(self.savepath, exist_ok=True)
        self.num_targets = num_targets
        self.loss_train = np.zeros((epochs, num_targets))
        self.mu_list = np.zeros((epochs, num_targets))
        self.rho_list = np.zeros((epochs, num_targets))

        self.train_norm = train_norm
        if self.train_norm is not None:
            self.train_norm.set_device(str(self.device))

    def train(self, epochs=100):
        self.net.train()
        for epoch in range(epochs):
            ELBO_samples = 5 if epoch == 0 else 1
            nb_samples = 0
            lastloss = 0

            for i, data in enumerate(self.train_loader):
                start_time = datetime.datetime.now()

                # Use non_blocking transfers (requires pin_memory=True in DataLoader)
                inputs, labels = data
                inputs = inputs.to(self.device, non_blocking=True).float()
                labels = labels.to(
                    self.device, non_blocking=True).float().unsqueeze(-1)

                overall_loss, losses, p_mu, p_rho = self.net.fit(
                    in_x=inputs, in_y=labels,
                    samples=ELBO_samples, scaler=self.train_norm, device=self.device)

                # Gather metrics from GPU (minimal transfer)
                self.mu_list[epoch] += p_mu.cpu().detach().numpy()
                self.rho_list[epoch] += p_rho.cpu().detach().numpy()
                for j in range(self.num_targets):
                    self.loss_train[epoch, j] += losses[j]
                nb_samples += inputs.size(0)

                end_time = datetime.datetime.now()
                elapsed = end_time - start_time
                eta = elapsed * (self.Nbatch - i) + elapsed * \
                    self.Nbatch * (epochs - epoch - 1)

                loss_str = " ".join(
                    [f"loss{j+1}: {losses[j].item():.4f}" for j in range(self.num_targets)])
                print(f"eta: {eta} epoch: {epoch+1:4d}/{epochs:4d}, batch: {i+1:5d} "
                      f"loss: {overall_loss.item():.4f} LossChange: {overall_loss.item() - lastloss:.4f} {loss_str}")
                lastloss = overall_loss.item()

                self._log_metrics(overall_loss, losses, p_mu, p_rho, epoch, i)

            self.loss_train[epoch] /= self.Nbatch
            self.mu_list[epoch] /= self.Nbatch
            self.rho_list[epoch] /= self.Nbatch
        return self.net

        if self.modelsave:
            try:
                self._save_model(self.savename)
            except Exception as e:
                print(f"Model saving failed: {str(e)}")

        print('Finished Training')

    def _log_metrics(self, overall_loss, losses, p_mu, p_rho, epoch, batch):
        # Optionally add tensorboard or logging code here.
        pass

    def _save_model(self, savename):
        torch.save(self.net, os.path.join(self.savepath, f"{savename}.pt"))
        np.save(os.path.join(self.savepath,
                f"{savename}_mu_list"), self.mu_list)
        np.save(os.path.join(self.savepath,
                f"{savename}_rho_list"), self.rho_list)
        np.save(os.path.join(self.savepath,
                f"{savename}_loss_train"), self.loss_train)

    def save_model(self, savepath='modelsave/bmdet/', savename='bmdet_best.pt'):
        torch.save(self.net, os.path.join(savepath, savename))
        np.save(os.path.join(savepath, "mu_list"), self.mu_list)
        np.save(os.path.join(savepath, "rho_list"), self.rho_list)
        np.save(os.path.join(savepath, "loss_train"), self.loss_train)

    def test(self, test_loader, samples=20):
        self.net.eval()
        metric_vals = []
        with torch.no_grad():
            for i, (x_test, y_test) in enumerate(test_loader):
                # If testing on CPU, move data non_blocking if possible.
                x_test = x_test.to(self.device, non_blocking=True)
                outs = self.net.test(
                    in_test=x_test, samples=samples, scaler=self.train_norm, force_cpu=True)
                metric_val = compute_metrics(
                    outs, y_test, train_scaler=self.train_norm)
                metric_vals.append(metric_val)

        return {
            "ACR": np.mean([m['avg_coverage_rate'] for m in metric_vals]),
            "AIL": np.mean([m['avg_interval_length'] for m in metric_vals]),
            "AES": np.mean([m['energy_score'] for m in metric_vals])
        }

# ---------------------------
# Multiprocess Training Functions
# ---------------------------


def train_on_gpu(model_class, trainer_class, train_loader, test_loader, train_norm,
                 params, param_idx, trainer_params, gpu_id, progress_file_path):
    """
    Train one model on the specified GPU and log progress.
    """
    device = torch.device(f"cuda:{gpu_id}")
    # Log the configuration early to help debug parameter issues.
    with open(progress_file_path, 'a') as f:
        f.write(
            f"[{datetime.datetime.now()}] START TRAINING: Model-{param_idx} on GPU {gpu_id}, params={params}\n")

    # Create and move model to device in one step.
    model = model_class(**params).to(device)
    trainer = trainer_class(
        model_wrapper=model,
        train_loader=train_loader,
        train_norm=train_norm,
        device=device
    )

    trainer.train(**trainer_params)
    with open(progress_file_path, 'a') as f:
        f.write(
            f"[{datetime.datetime.now()}] FINISH TRAINING: Model-{param_idx}\n")

    with open(progress_file_path, 'a') as f:
        f.write(f"[{datetime.datetime.now()}] START TESTING: Model-{param_idx}\n")

    metrics = trainer.test(test_loader=test_loader, samples=100)
    with open(progress_file_path, 'a') as f:
        f.write(
            f"[{datetime.datetime.now()}] FINISH TESTING: Model-{param_idx}\n")

    # Clean up GPU memory: move network to CPU and freeze parameters.
    trainer.net.to('cpu')
    for p in trainer.net.parameters():
        p.requires_grad = False
    model_sd = trainer.net.state_dict()
    # Clear GPU memory
    torch.cuda.empty_cache()
    with torch.cuda.device(f"cuda:{gpu_id}"):
        torch.cuda.empty_cache()
    return metrics, model_sd


def grid_search_torch_model(model_class: nn.Module,
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
                            test_norm=None,
                            max_workers=2):

    param_combinations = list(itertools.product(*param_grid.values()))
    n_models = len(param_combinations)
    os.makedirs(savedir, exist_ok=True)
    progress_file_path = os.path.join(savedir, "progress_log.txt")
    with open(progress_file_path, 'w') as pf:
        pf.write(f"Total number of models to train: {n_models}\n")

    futures = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_grid.keys(), params))
            gpu_id = i % max_workers
            future = executor.submit(
                train_on_gpu,
                model_class,
                trainer_class,
                train_loader,
                test_loader,
                train_norm,
                param_dict,
                i,
                training_args,
                gpu_id,
                progress_file_path
            )
            futures.append((param_dict, future))

        results = []
        for params, f in futures:
            metrics, model_sd = f.result()
            results.append((params, metrics, model_sd))

    # Pick best model based on difference from ACR=0.8
    acr_diffs = np.array([math.fabs(r[1]['ACR'] - 0.8) for r in results])
    best_diff_idx = np.argmin(acr_diffs)
    best_weights = results[best_diff_idx][2]
    best_params = results[best_diff_idx][0]
    best_metrics = results[best_diff_idx][1]

    try:
        if best_weights is not None:
            torch.save(best_weights, os.path.join(savedir, savename))
        else:
            print('Best model NOT saved :(')
    except Exception as e:
        print('Best model NOT saved :(', e)

    try:
        if best_params is not None:
            with open(os.path.join(savedir, 'best_hyperparams.json'), 'w') as f:
                json.dump(best_params, f)
    except Exception as e:
        print("Best hyperparams not saved.", e)

    try:
        if best_metrics is not None:
            with open(os.path.join(savedir, 'best_test_metrics.json'), 'w') as f:
                json.dump(best_metrics, f)
    except Exception as e:
        print("Best metrics not saved.", e)
