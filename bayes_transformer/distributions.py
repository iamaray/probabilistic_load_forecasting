import torch
import numpy as np
import torch.nn as nn


class TrainableRandomDistribution(nn.Module):
    # Samples for variational inference as in Weights Uncertainity on Neural Networks (Bayes by backprop paper)

    def __init__(self, mu, rho):
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi

    def sample(self):
        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w

    def log_posterior(self, w=None):
        assert (self.w is not None)
        if w is None:
            w = self.w

        log_sqrt2pi = np.log(np.sqrt(2*self.pi))
        log_posteriors = -log_sqrt2pi - \
            torch.log(self.sigma) - (((w - self.mu) ** 2) /
                                     (2 * self.sigma ** 2)) - 0.5
        return log_posteriors.sum()


class PriorWeightDistribution(nn.Module):
    def __init__(self,
                 pi=1,
                 sigma1=0.1,
                 sigma2=0.001,
                 dist=None,
                 use_mixture=False):
        super(PriorWeightDistribution, self).__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dist1 = None
        self.dist2 = None
        self.custom_dist = dist

        # If no custom prior is provided, use a Gaussian mixture prior.
        if dist is None:
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0, sigma1)
            self.dist2 = torch.distributions.Normal(0, sigma2)
            self.use_mixture = True
        else:
            self.custom_dist = dist
            self.use_mixture = False

    def __getattr__(self, name):
        if name == 'use_custom_dist':
            return True if self.pi != -1 else False
        if name == 'use_mixture':
            return False if self.use_custom_dist else True
        return super().__getattr__(name)

    def log_prior(self, w):
        if not self.use_mixture:
            return self.custom_dist.log_prob(w).sum()
        prob_n1 = torch.exp(self.dist1.log_prob(w))
        prob_n2 = torch.exp(self.dist2.log_prob(w))
        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2) + 1e-6
        return (torch.log(prior_pdf) - 0.5).sum()


class PriorWeightStudentT(nn.Module):
    def __init__(self, nu=3, mu=0.0, sigma=1.0):
        super(PriorWeightStudentT, self).__init__()
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.dist = torch.distributions.StudentT(df=nu, loc=mu, scale=sigma)

    def log_prior(self, w):
        return self.dist.log_prob(w).sum()


# class PriorWeight
