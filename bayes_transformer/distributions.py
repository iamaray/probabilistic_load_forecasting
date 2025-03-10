import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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

    def to(self, device):
        """Move all tensors to the specified device"""
        # Update the device attribute
        self.device = device

        # Move custom distribution to device if it exists
        if self.custom_dist is not None and hasattr(self.custom_dist, 'to'):
            try:
                self.custom_dist = self.custom_dist.to(device)
            except Exception as e:
                print(f"Warning: Could not move custom_dist to {device}: {e}")

        # Call the parent class's to() method
        return super().to(device)


class PriorWeightDistribution(nn.Module):
    def __init__(self,
                 pi=1,
                 sigma1=0.1,
                 sigma2=0.001,
                 dist=None,
                 use_mixture=False,
                 device=None):
        super(PriorWeightDistribution, self).__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dist1 = None
        self.dist2 = None
        self.custom_dist = dist
        self.device = device

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

            # Move custom distribution to device if specified
            if device is not None and hasattr(self.custom_dist, 'to'):
                self.custom_dist.to(device)

    def __getattr__(self, name):
        if name == 'custom_dist':
            return None
        if name == 'use_mixture':
            return False if self.custom_dist is not None else True
        return super().__getattr__(name)

    def log_prior(self, w):
        # Move custom distribution to data's device if needed
        if not self.use_mixture and hasattr(self.custom_dist, 'to'):
            if hasattr(self.custom_dist, 'get_param_device'):
                if w.device != self.custom_dist.get_param_device():
                    self.custom_dist.to(w.device)
            elif self.device != w.device:
                self.custom_dist.to(w.device)
                self.device = w.device

        if not self.use_mixture:
            return self.custom_dist.log_prob(w).sum()
        prob_n1 = torch.exp(self.dist1.log_prob(w))
        prob_n2 = torch.exp(self.dist2.log_prob(w))
        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2) + 1e-6
        return (torch.log(prior_pdf) - 0.5).sum()

    def to(self, device):
        """Move all tensors to the specified device"""
        # Update the device attribute
        self.device = device

        # Move custom distribution to device if it exists
        if self.custom_dist is not None and hasattr(self.custom_dist, 'to'):
            try:
                self.custom_dist = self.custom_dist.to(device)
            except Exception as e:
                print(f"Warning: Could not move custom_dist to {device}: {e}")

        # Call the parent class's to() method
        return super().to(device)


class PriorWeightDistributionTemplate(nn.Module):
    def __init__(self, device=None):
        super(PriorWeightDistributionTemplate, self).__init__()
        self.device = device

    def log_prior(self, w):
        # Move distribution parameters to the data's device if needed
        if w.device != self.get_param_device():
            self.to(w.device)
        raise NotImplementedError

    def fit_params_to_data(self, data: torch.Tensor):
        # Move distribution parameters to the data's device if needed
        if data.device != self.get_param_device():
            data = data.to(self.device)
        raise NotImplementedError

    def get_param_device(self):
        """Get the device of the distribution parameters"""
        # This should be overridden by subclasses to return the device of a parameter tensor
        return self.device if self.device is not None else 'cpu'

    def to(self, device):
        """Move all tensors to the specified device"""
        self.device = device

        # Move all tensor attributes to the device
        if hasattr(self, 'PI') and self.PI is not None:
            self.PI = self.PI.to(device)
        if hasattr(self, 'SIGMA') and self.SIGMA is not None:
            self.SIGMA = self.SIGMA.to(device)
        if hasattr(self, 'MU') and self.MU is not None:
            self.MU = self.MU.to(device)

        # Recreate distributions with tensors on the new device
        if hasattr(self, 'dists'):
            self.dists = [torch.distributions.Normal(
                loc=self.MU[i].to(device),
                scale=self.SIGMA[i].to(device))
                for i in range(len(self.MU))]

        return super().to(device)


class PriorWeightGMM(PriorWeightDistributionTemplate):
    def __init__(self, proportions: torch.Tensor = None, stds: torch.Tensor = None, locs: torch.Tensor = None, device=None, n=None):
        super(PriorWeightGMM, self).__init__(device=device)

        if proportions is None:
            assert n is not None
            proportions = torch.softmax(torch.ones(n, device=device), dim=0)

        assert proportions.sum() == 1

        # Move tensors to specified device if provided
        if device is not None:
            if proportions is not None:
                proportions = proportions.to(device)
            if stds is None:
                assert n is not None
                stds = torch.ones(n, device=device)
            else:
                stds = stds.to(device)
            if locs is None:
                locs = torch.zeros(n, device=device)
            else:
                locs = locs.to(device)

        self.PI = proportions
        self.SIGMA = stds
        self.N = len(proportions) if n is None else n

        if locs is None:
            locs = torch.Tensor([0.0 for _ in range(self.N)]).to(
                self.get_param_device())

        self.MU = locs

        self.dists = [torch.distributions.Normal(
            loc=locs[i], scale=stds[i]) for i in range(self.N)]

    def log_prior(self, w):
        # Move distribution parameters to the data's device if needed
        if w.device != self.get_param_device():
            self.to(w.device)

        # Calculate log probabilities for each component
        log_probs = torch.stack([torch.log(self.PI[i]) + self.dists[i].log_prob(w)
                                for i in range(self.N)])

        max_log_prob = torch.max(log_probs, dim=0)[0]
        log_sum = max_log_prob + torch.log(torch.sum(
            torch.exp(log_probs - max_log_prob.unsqueeze(0)), dim=0))

        return log_sum.sum()

    def fit_params_to_data(self, data):
        # Ensure data is on the same device as the distribution parameters
        if self.device is not None and data.device != self.device:
            data = data.to(self.device)

        # Store original parameters for reporting
        orig_mu = self.MU.clone()
        orig_sigma = self.SIGMA.clone()

        print(f"GMM fitting - Before: MU={orig_mu}, SIGMA={orig_sigma}")

        # Compute empirical mean and std
        mean = data.mean()
        std = data.std()

        # For simplicity, adjust the means and stds while keeping proportions fixed
        self.MU = torch.tensor(
            [mean + i*std/self.N for i in range(self.N)], device=self.get_param_device())
        self.SIGMA = torch.tensor(
            [std for _ in range(self.N)], device=self.get_param_device())

        # Update distributions
        self.dists = [torch.distributions.Normal(
            loc=self.MU[i], scale=self.SIGMA[i]) for i in range(self.N)]

        print(f"GMM fitting - After: MU={self.MU}, SIGMA={self.SIGMA}")
        print(f"Data stats: mean={mean.item():.4f}, std={std.item():.4f}")

        return self

    def get_param_device(self):
        """Get the device of the distribution parameters"""
        return self.PI.device

    def to(self, device):
        """Move all tensors to the specified device and recreate distributions"""
        # Move all tensor attributes to the device
        if hasattr(self, 'PI') and self.PI is not None:
            self.PI = self.PI.to(device)

        if hasattr(self, 'SIGMA') and self.SIGMA is not None:
            self.SIGMA = self.SIGMA.to(device)

        if hasattr(self, 'MU') and self.MU is not None:
            self.MU = self.MU.to(device)

        # Update the device attribute
        self.device = device

        # Recreate the distributions with tensors on the new device
        if hasattr(self, 'dists') and hasattr(self, 'N'):
            self.dists = [torch.distributions.Normal(
                loc=self.MU[i],
                scale=self.SIGMA[i])
                for i in range(self.N)]

        # Call the parent class's to() method
        return super().to(device)


class PriorWeightGaussian(PriorWeightGMM):
    def __init__(self, std: float, device=None):
        super().__init__(
            proportions=torch.tensor([1.0], device=device),
            stds=torch.tensor([std], device=device),
            device=device
        )

    def to(self, device):
        """Move all tensors to the specified device"""
        # Use the parent class's to() method since it handles all the tensors
        return super().to(device)

    def fit_params_to_data(self, data):
        print(
            f"Gaussian fitting - Before: mu={self.MU.item()}, sigma={self.SIGMA.item()}")
        result = super().fit_params_to_data(data)
        print(
            f"Gaussian fitting - After: mu={self.MU.item()}, sigma={self.SIGMA.item()}")
        return result


class PriorWeightTMM(PriorWeightDistributionTemplate):
    def __init__(self, proportions: torch.Tensor = None, nus: torch.Tensor = None, sigmas: torch.Tensor = None,
                 mus: torch.Tensor = None, device=None, n=None):
        super().__init__(device=device)
        if proportions is None:
            assert n is not None
            proportions = torch.softmax(torch.ones(n, device=device), dim=0)

        assert proportions.sum() == 1

        # Move tensors to specified device if provided
        if device is not None:
            if proportions is not None:
                proportions = proportions.to(device)
            if nus is None:
                assert n is not None
                nus = torch.ones(n, device=device) * 3
            else:
                nus = nus.to(device)
            if sigmas is None:
                assert n is not None
                sigmas = torch.ones(n, device=device)
            else:
                sigmas = sigmas.to(device)
            if mus is not None:
                mus = mus.to(device)

        self.PI = proportions
        self.NU = nus
        self.SIGMA = sigmas
        self.MU = torch.zeros_like(nus) if mus is None else mus
        self.N = len(proportions) if n is None else n

        self.dists = [torch.distributions.StudentT(
            df=nus[i], loc=self.MU[i], scale=sigmas[i]) for i in range(self.N)]

    def log_prior(self, w):
        # Move distribution parameters to the data's device if needed
        if w.device != self.get_param_device():
            self.to(w.device)

        log_probs = torch.stack([torch.log(self.PI[i]) + self.dists[i].log_prob(w)
                                for i in range(self.N)])

        max_log_prob = torch.max(log_probs, dim=0)[0]
        log_sum = max_log_prob + torch.log(torch.sum(
            torch.exp(log_probs - max_log_prob.unsqueeze(0)), dim=0))

        return log_sum.sum()

    def fit_params_to_data(self, data):
        # Ensure data is on the same device as the distribution parameters
        if self.device is not None and data.device != self.device:
            data = data.to(self.device)

        # Store original parameters for reporting
        orig_mu = self.MU.clone()
        orig_sigma = self.SIGMA.clone()

        print(
            f"TMM fitting - Before: MU={orig_mu}, SIGMA={orig_sigma}, NU={self.NU}")

        # Compute empirical statistics
        mean = data.mean()
        std = data.std()

        # For each component, adjust location and scale
        for i in range(self.N):
            # Adjust location parameter
            self.MU[i] = mean + i*std/self.N

            # Adjust scale parameter based on degrees of freedom
            nu = self.NU[i]
            if nu > 2:
                self.SIGMA[i] = std * torch.sqrt((nu-2)/nu)
            else:
                self.SIGMA[i] = std

        # Update distributions
        self.dists = [torch.distributions.StudentT(
            df=self.NU[i], loc=self.MU[i], scale=self.SIGMA[i]) for i in range(self.N)]

        print(
            f"TMM fitting - After: MU={self.MU}, SIGMA={self.SIGMA}, NU={self.NU}")
        print(f"Data stats: mean={mean.item():.4f}, std={std.item():.4f}")

        return self

    def get_param_device(self):
        """Get the device of the distribution parameters"""
        return self.PI.device

    def to(self, device):
        """Move all tensors to the specified device and recreate distributions"""

        if hasattr(self, 'PI') and self.PI is not None:
            self.PI = self.PI.to(device)
        if hasattr(self, 'NU') and self.NU is not None:
            self.NU = self.NU.to(device)
        if hasattr(self, 'SIGMA') and self.SIGMA is not None:
            self.SIGMA = self.SIGMA.to(device)
        if hasattr(self, 'MU') and self.MU is not None:
            self.MU = self.MU.to(device)

        self.device = device

        if hasattr(self, 'dists') and hasattr(self, 'N'):
            self.dists = [torch.distributions.StudentT(
                df=self.NU[i],
                loc=self.MU[i],
                scale=self.SIGMA[i])
                for i in range(self.N)]

        return super().to(device)


class PriorWeightStudentT(PriorWeightTMM):
    def __init__(self, nu=3, mu=0.0, sigma=1.0, device=None):
        super().__init__(
            proportions=torch.tensor([1.0], device=device),
            nus=torch.tensor([nu], device=device),
            sigmas=torch.tensor([sigma], device=device),
            mus=torch.tensor([mu], device=device),
            device=device
        )

    def log_prior(self, w):
        return super().log_prior(w)

    def fit_params_to_data(self, data):
        print(
            f"StudentT fitting - Before: nu={self.NU.item()}, mu={self.MU.item()}, sigma={self.SIGMA.item()}")
        result = super().fit_params_to_data(data)
        print(
            f"StudentT fitting - After: nu={self.NU.item()}, mu={self.MU.item()}, sigma={self.SIGMA.item()}")
        return result

    def get_param_device(self):
        return super().get_param_device()

    def to(self, device):
        """Move all tensors to the specified device"""
        # Use the parent class's to() method since it handles all the tensors
        return super().to(device)


class PriorWeightPhaseType(PriorWeightDistributionTemplate):
    def __init__(self, S: torch.Tensor = None, alpha: torch.Tensor = None, n: int = None, device=None):
        super().__init__(device=device)

        # Handle device placement
        if device is not None:
            if S is not None:
                S = S.to(device)
            if alpha is not None:
                alpha = alpha.to(device)

        self.S = S  # Sub-intensity matrix
        if S is None:
            assert n is not None
            self.S = torch.tril(torch.randn(
                (n, n), device=device if device is not None else 'cpu'))
        else:
            assert S.shape[-1] == S.shape[-2]

        self.alpha = alpha  # Initial distribution
        if alpha is None:
            n = self.S.shape[0]
            self.alpha = torch.softmax(
                torch.randn(n, device=self.get_param_device()), dim=0)
        else:
            assert alpha.shape[-1] == S.shape[-1]
            assert alpha.sum() == 1

        self.s = -self.S.sum(dim=1)  # Exit rates vector

    def log_prior(self, w):
        # Move distribution parameters to the data's device if needed
        if w.device != self.get_param_device():
            self.to(w.device)

        w_flat = w.flatten()
        log_probs = []

        for wi in w_flat:
            exp_Sw = torch.matrix_exp(self.S * wi)
            density = self.alpha @ exp_Sw @ self.s
            log_probs.append(torch.log(density))

        log_probs = torch.stack(log_probs)
        return log_probs.sum()

    def fit_params_to_data(self, data: torch.Tensor):
        # Ensure data is on the same device as the distribution parameters
        device = self.get_param_device()
        if data.device != device:
            data = data.to(device)

        data = F.softplus(data)

        # Store original parameters for reporting
        orig_S = self.S.clone()
        orig_alpha = self.alpha.clone()

        print(f"PhaseType fitting - Before:")
        print(f"S matrix (first few values): {orig_S.flatten()[:5]}")
        print(f"alpha vector: {orig_alpha}")

        # Filter out zero values and take absolute values for phase-type fitting
        data = torch.abs(data.flatten())
        data = data[data > 0]

        # Check if we have any positive data points left
        if data.numel() == 0:
            raise ValueError(
                "No positive values in data after filtering zeros")

        n = self.S.shape[0]  # Number of phases

        # Initialize parameters if not already set
        if not hasattr(self, 'S') or not hasattr(self, 'alpha'):
            # Lower triangular structure for S
            self.S = torch.tril(torch.randn(n, n, device=device))
            self.alpha = torch.softmax(torch.randn(n, device=device), dim=0)
            self.s = -self.S.sum(dim=1)

        # Sort data for numerical stability
        data, _ = torch.sort(data)

        print(
            f"Data stats: mean={data.mean().item():.4f}, std={data.std().item():.4f}, min={data.min().item():.4f}, max={data.max().item():.4f}")
        print(
            f"Fitting phase-type distribution with {n} phases to {len(data)} data points")

        # EM algorithm for phase-type distribution fitting
        max_iter = 20
        tolerance = 1e-4
        prev_log_likelihood = -float('inf')

        # Create identity matrix for calculations
        I = torch.eye(n, device=device)

        # Pre-compute regularization matrix
        reg_eye = torch.eye(n, device=device) * 1e-6

        # Batch processing parameters
        batch_size = min(100, len(data))  # Process data in batches

        for iteration in range(max_iter):
            # E-step: Compute expected sufficient statistics
            B = torch.zeros((n, n), device=device)
            z = torch.zeros(n, device=device)
            print(f"E-step: {iteration}")

            log_likelihood = 0.0

            # Process data in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]

                # Vectorized computation of matrix exponentials
                exp_Sx_batch = torch.stack(
                    [torch.matrix_exp(self.S * x) for x in batch])
                print
                # Compute densities and log-likelihoods for the batch
                densities = torch.stack(
                    [self.alpha @ exp_Sx @ self.s for exp_Sx in exp_Sx_batch])
                log_likelihood += torch.log(densities).sum()

                # Compute conditional expectations for the batch
                reg_matrix = -self.S + reg_eye

                # Try solving the linear system for all observations in the batch
                try:
                    # Vectorized computation for E_i
                    E_i_batch = torch.stack([
                        self.alpha @ torch.linalg.solve(reg_matrix, I - exp_Sx)
                        for exp_Sx in exp_Sx_batch
                    ])
                except torch._C._LinAlgError:
                    # Fallback to pseudo-inverse if needed
                    reg_matrix_pinv = torch.linalg.pinv(reg_matrix)
                    E_i_batch = torch.stack([
                        self.alpha @ reg_matrix_pinv @ (I - exp_Sx)
                        for exp_Sx in exp_Sx_batch
                    ])

                # Update sufficient statistics for the batch
                for idx, (x, exp_Sx, density) in enumerate(zip(batch, exp_Sx_batch, densities)):
                    E_i = E_i_batch[idx]

                    # Compute E_ij more efficiently using vectorized operations where possible
                    # For transitions with non-zero rates
                    mask = (torch.eye(n, device=device) == 0) & (self.S != 0)
                    if mask.any():
                        # Use fewer time points for integration to improve efficiency
                        t_points = torch.linspace(
                            0, x.item(), 50, device=device)

                        # Compute all matrix exponentials at once
                        exp_St = torch.stack(
                            [torch.matrix_exp(self.S * t) for t in t_points])

                        # For each non-zero transition
                        indices = mask.nonzero(as_tuple=True)
                        for idx in range(len(indices[0])):
                            i, j = indices[0][idx].item(
                            ), indices[1][idx].item()
                            exp_values = exp_St[:, i, j]
                            B[i, j] += self.alpha[i] * self.S[i, j] * \
                                torch.trapz(exp_values, t_points)

                    # Update z
                    z += self.alpha * exp_Sx @ self.s / density

            # Print log likelihood for debugging
            print(
                f'Iteration {iteration}, log_likelihood: {log_likelihood.item():.4f}')

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                break
            prev_log_likelihood = log_likelihood

            # M-step: Update parameters
            # Update initial distribution
            self.alpha = z / len(data)
            self.alpha = self.alpha / self.alpha.sum()  # Normalize

            # Update transition rate matrix more efficiently
            for i in range(n):
                total_time_i = torch.sum(E_i_batch[:, i])
                if total_time_i > 0:
                    # Update all transitions from state i at once
                    mask = torch.arange(n, device=device) != i
                    self.S[i, mask] = B[i, mask] / total_time_i

            # Ensure diagonal elements make row sums zero (vectorized)
            row_sums = torch.sum(self.S, dim=1)
            self.S.diagonal().copy_(-row_sums + self.S.diagonal())

            # Ensure diagonal elements are negative
            diag_mask = self.S.diagonal() >= 0
            if diag_mask.any():
                self.S.diagonal()[diag_mask] = -0.1

            # Recompute exit rates
            self.s = -self.S.sum(dim=1)

        # Ensure numerical stability
        # Make sure S is lower triangular with negative diagonal
        self.S = torch.tril(self.S)
        diag_mask = self.S.diagonal() >= 0
        if diag_mask.any():
            self.S.diagonal()[diag_mask] = -0.1

        # Recompute exit rates
        self.s = -self.S.sum(dim=1)

        # Normalize alpha
        self.alpha = self.alpha / self.alpha.sum()

        # After fitting is complete
        print(f"PhaseType fitting - After:")
        print(f"S matrix (first few values): {self.S.flatten()[:5]}")
        print(f"alpha vector: {self.alpha}")

        # Calculate the mean of the fitted distribution for comparison
        try:
            fitted_mean = self.alpha @ torch.linalg.inv(-self.S) @ torch.ones(
                n, device=device)
            print(
                f"Fitted distribution mean: {fitted_mean.item():.4f} (Data mean: {data.mean().item():.4f})")
        except:
            print("Could not calculate fitted distribution mean")

        return self

    def get_param_device(self):
        """Get the device of the distribution parameters"""
        return self.S.device

    def to(self, device):
        """Move all tensors to the specified device"""
        # Move all tensor attributes to the device
        if hasattr(self, 'S') and self.S is not None:
            self.S = self.S.to(device)

        if hasattr(self, 'alpha') and self.alpha is not None:
            self.alpha = self.alpha.to(device)

        if hasattr(self, 's') and self.s is not None:
            self.s = self.s.to(device)

        # Update the device attribute
        self.device = device

        # Call the parent class's to() method
        return super().to(device)

# Cauchy distribution from T distribution


class PriorWeightCauchyMM(PriorWeightTMM):
    def __init__(self, proportions: torch.Tensor = None, mus: torch.Tensor = None, sigmas: torch.Tensor = None, device=None, n=None):
        super().__init__(
            proportions=proportions,
            nus=torch.ones(n, device=device),
            sigmas=sigmas,
            mus=mus,
            device=device,
            n=n
        )

    def log_prior(self, w):
        return super().log_prior(w)

    def fit_params_to_data(self, data):
        print(f"CauchyMM fitting - Before: mu={self.MU}, sigma={self.SIGMA}")
        result = super().fit_params_to_data(data)
        print(f"CauchyMM fitting - After: mu={self.MU}, sigma={self.SIGMA}")
        return result

    def get_param_device(self):
        return super().get_param_device()

    def to(self, device):
        """Move all tensors to the specified device"""
        # Use the parent class's to() method since it handles all the tensors
        return super().to(device)
