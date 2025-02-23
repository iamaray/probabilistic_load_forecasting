import numpy as np

from .layers import BayesianLinear
import torch
import torch.nn as nn
####################


def rmse_loss(true, pre):
    criterion = torch.nn.MSELoss(reduction="mean")
    RMSELoss = torch.sqrt(criterion(pre, true))
    return RMSELoss


class taskbalance(nn.Module):
    def __init__(self, num=3):
        super(taskbalance, self).__init__()
        self.num = num
        w_mu = torch.Tensor([0.5 for _ in range(num)])
        w_rho = torch.Tensor([0.01 for _ in range(num)])

        self.weight_mu = nn.Parameter(w_mu)
        self.weight_rho = nn.Parameter(w_rho)

    def forward(self, losses, device_str='cpu'):
        self.device = torch.device(device_str)
        stds = torch.stack([torch.randn(1).to(self.device)
                            for _ in range(self.num)])

        self.weight_mu = self.weight_mu.to(self.device)
        self.weight_rho = self.weight_rho.to(self.device)
        W = stds * self.weight_rho + self.weight_mu

        loss = torch.tensor(0.0).to(self.device)

        print("")
        print("weight_mu", self.weight_mu.data)
        print("weight_rho", self.weight_rho.data)

        for i in range(self.num):
            loss = loss + 0.5 / (W[i] ** 4) * losses[i] + torch.log(W[i] ** 2)
            print(f"sample sigma_{i}:", W[i]**2)
            print("")

        return loss


def variational_estimator(nn_class):
    def nn_kl_divergence(self):
        kl_divergence = 0
        for module in self.modules():
            if isinstance(module, (BayesianLinear)):
                kl_divergence += module.log_variational_posterior - module.log_prior
        return kl_divergence
    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)

    def sample_elbo_m(self, inputs, labels, num_targets, sample_nbr, scaler):
        losses = torch.zeros(num_targets).to(inputs.device)

        for _ in range(sample_nbr):
            outs = self(inputs)
            # scaler.reverse(outs)

            rmses = torch.stack([rmse_loss(outs[:, :, i], labels[:, :, i])
                                 for i in range(num_targets)])
            losses = losses + rmses

        kl = self.nn_kl_divergence()
        kl = kl * 1e-5
        return (losses / sample_nbr), kl
    setattr(nn_class, "sample_elbo_m", sample_elbo_m)

    return nn_class


def loadDataforTrainVal(input_size, output_size):
    pass
