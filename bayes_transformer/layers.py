import torch
from torch import nn
from torch.nn import functional as F
from .distributions import TrainableRandomDistribution, PriorWeightDistribution


class BayesianLinear(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1=0.1,
                 prior_sigma_2=0.4,
                 prior_pi=1,
                 posterior_mu_init=0,
                 posterior_rho_init=-7.0,
                 prior_dist=None):
        super().__init__()

        # init parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        self.weight_mu = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(posterior_rho_init, 0.1))

        self.weight_sampler = TrainableRandomDistribution(
            self.weight_mu, self.weight_rho)

        self.bias_mu = nn.Parameter(torch.Tensor(
            out_features).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(
            out_features).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = TrainableRandomDistribution(
            self.bias_mu, self.bias_rho)

        self.weight_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.bias_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features), device=x.device)
            b_log_posterior = 0
            b_log_prior = 0

        self.log_variational_posterior = self.weight_sampler.log_posterior() + \
            b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.linear(x, w, b)
