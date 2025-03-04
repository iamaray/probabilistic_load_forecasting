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
        if name == 'custom_dist':
            return None
        if name == 'use_mixture':
            return False if self.custom_dist is not None else True
        return super().__getattr__(name)

    def log_prior(self, w):
        if not self.use_mixture:
            return self.custom_dist.log_prob(w).sum()
        prob_n1 = torch.exp(self.dist1.log_prob(w))
        prob_n2 = torch.exp(self.dist2.log_prob(w))
        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2) + 1e-6
        return (torch.log(prior_pdf) - 0.5).sum()


class PriorWeightDistributionTemplate(nn.Module):
    def __init__(self):
        super(PriorWeightDistributionTemplate, self).__init__()

    def log_prior(self, w):
        raise NotImplementedError

    def fit_params_to_data(self, data: torch.Tensor):
        raise NotImplementedError


class PriorWeightGMM(PriorWeightDistributionTemplate):
    def __init__(self, proportions: torch.Tensor, stds: torch.Tensor, locs: torch.Tensor = None):
        super(PriorWeightGMM, self).__init__()
        assert proportions.sum() == 1

        self.PI = proportions
        self.SIGMA = stds
        self.N = len(proportions)

        if locs is None:
            locs = torch.Tensor([0.0 for _ in range(self.N)])

        self.MU = locs

        self.dists = [torch.distributions.Normal(
            loc=locs[i], scale=stds[i]) for i in range(self.N)]

    def log_prior(self, w):
        # Calculate log probabilities for each component, including mixing proportions
        # Each component's probability is: pi_i * N(w | mu_i, sigma_i)
        # In log space: log(pi_i) + log(N(w | mu_i, sigma_i))
        log_probs = torch.stack([torch.log(self.PI[i]) + self.dists[i].log_prob(w)
                                for i in range(self.N)])

        # Use log-sum-exp trick for numerical stability:
        # log(sum(exp(x_i))) = a + log(sum(exp(x_i - a))) where a = max(x_i)
        max_log_prob = torch.max(log_probs, dim=0)[0]
        log_sum = max_log_prob + torch.log(torch.sum(
            torch.exp(log_probs - max_log_prob.unsqueeze(0)), dim=0))

        return log_sum.sum()

    def fit_params_to_data(self, data):
        """
        Fit GMM parameters using simple moment matching
        """
        # Compute empirical mean and std
        mean = data.mean()
        std = data.std()

        # For simplicity, adjust the means and stds while keeping proportions fixed
        self.MU = torch.tensor([mean + i*std/self.N for i in range(self.N)])
        self.SIGMA = torch.tensor([std for _ in range(self.N)])

        # Update distributions
        self.dists = [torch.distributions.Normal(
            loc=self.MU[i], scale=self.SIGMA[i]) for i in range(self.N)]


class PriorWeightGaussian(PriorWeightGMM):
    def __init__(self, std: float):
        super().__init__(
            proportions=torch.tensor([1.0]),
            stds=torch.tensor([std])
        )


class PriorWeightTMM(PriorWeightDistributionTemplate):
    def __init__(self, proportions: torch.Tensor, nus: torch.Tensor, sigmas: torch.Tensor, mus: torch.Tensor = None):
        super().__init__()
        assert proportions.sum() == 1

        self.PI = proportions
        self.NU = nus
        self.SIGMA = sigmas
        self.MU = torch.zeros_like(nus) if mus is None else mus
        self.N = len(proportions)

        self.dists = [torch.distributions.StudentT(
            df=nus[i], loc=self.MU[i], scale=sigmas[i]) for i in range(self.N)]

    def log_prior(self, w):
        log_probs = torch.stack([torch.log(self.PI[i]) + self.dists[i].log_prob(w)
                                for i in range(self.N)])

        max_log_prob = torch.max(log_probs, dim=0)[0]
        log_sum = max_log_prob + torch.log(torch.sum(
            torch.exp(log_probs - max_log_prob.unsqueeze(0)), dim=0))

        return log_sum.sum()

    def fit_params_to_data(self, data):
        """
        Fit Student-t mixture parameters using moment matching
        """
        # Compute empirical statistics
        mean = data.mean()
        std = data.std()

        # For each component, adjust location and scale
        for i in range(self.N):
            # Adjust location parameter
            self.MU[i] = mean + i*std/self.N

            # Adjust scale parameter based on degrees of freedom
            # For Student-t, variance = (nu/(nu-2))*sigma^2 when nu > 2
            # So sigma = std * sqrt((nu-2)/nu)
            nu = self.NU[i]
            if nu > 2:
                self.SIGMA[i] = std * torch.sqrt((nu-2)/nu)
            else:
                self.SIGMA[i] = std

        # Update distributions
        self.dists = [torch.distributions.StudentT(
            df=self.NU[i], loc=self.MU[i], scale=self.SIGMA[i]) for i in range(self.N)]


class PriorWeightStudentT(PriorWeightTMM):
    def __init__(self, nu=3, mu=0.0, sigma=1.0):
        super().__init__(
            proportions=torch.tensor([1.0]),
            nus=torch.tensor([nu]),
            sigmas=torch.tensor([sigma]),
            mus=torch.tensor([mu])
        )


class PriorWeightPhaseType(PriorWeightDistributionTemplate):
    def __init__(self, S: torch.Tensor, alpha: torch.Tensor):
        super().__init__()
        self.S = S  # Sub-intensity matrix
        assert S.shape[-1] == S.shape[-2]

        self.alpha = alpha  # Initial distribution
        assert alpha.sum() == 1

        self.s = -S.sum(dim=1)  # Exit rates vector

    def log_prior(self, w):
        w_flat = w.flatten()
        log_probs = []

        for wi in w_flat:
            exp_Sw = torch.matrix_exp(self.S * wi)
            density = self.alpha @ exp_Sw @ self.s
            log_probs.append(torch.log(density))

        log_probs = torch.stack(log_probs)
        return log_probs.sum()

    def fit_params_to_data(self, data: torch.Tensor):
        """
        Fit phase-type distribution parameters (S matrix and alpha vector) to data using EM algorithm.

        Args:
            data: Tensor of observations to fit the distribution to

        Note: This is a simplified implementation - a full EM algorithm for phase-type
        distributions is quite complex and computationally intensive.
        """
        # Ensure data is positive
        if (data <= 0).any():
            raise ValueError(
                "Phase-type distributions are only defined for positive values")

        n = self.S.shape[0]  # Number of phases

        # Initialize parameters if not already set
        if not hasattr(self, 'S') or not hasattr(self, 'alpha'):
            # Lower triangular structure
            self.S = torch.tril(torch.randn(n, n))
            self.alpha = torch.softmax(torch.randn(n), dim=0)

        # Simple moment matching for mean
        empirical_mean = data.mean()

        # For phase-type distributions:
        # mean = alpha @ (-S)^(-1) @ 1
        # We can use this to scale S appropriately
        current_mean = self.alpha @ torch.linalg.inv(-self.S) @ torch.ones(n)
        scale_factor = empirical_mean / current_mean

        # Scale S to match the empirical mean
        self.S = self.S / scale_factor

        # Recompute exit rates
        self.s = -self.S.sum(dim=1)

    # class PriorWeightStudentT(nn.Module):
    #     def __init__(self, nu=3, mu=0.0, sigma=1.0):
    #         super(PriorWeightStudentT, self).__init__()
    #         self.nu = nu
    #         self.mu = mu
    #         self.sigma = sigma
    #         self.dist = torch.distributions.StudentT(df=nu, loc=mu, scale=sigma)

    #     def log_prior(self, w):
    #         return self.dist.log_prob(w).sum()
