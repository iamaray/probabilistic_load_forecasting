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
        # Move distribution parameters to the data's device if needed
        if w.device != self.get_param_device():
            self.to(w.device)

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

        # CUDA optimization: Set non-blocking transfers for better parallelism
        torch.cuda.set_device(self.device if self.device != 'cpu' else 0)

        # CUDA optimization: Enable tensor cores for faster matrix operations if available
        if self.device != 'cpu' and torch.cuda.is_available():
            # Check if we're using a GPU that supports tensor cores
            if torch.cuda.get_device_capability(self.device.index if hasattr(self.device, 'index') else 0)[0] >= 7:
                # Enable TF32 precision on Ampere or newer GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("CUDA optimization: TF32 precision enabled for matrix operations")

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
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        super().__init__(device=self.device)

        if device is not None:
            if S is not None:
                S = S.to(self.device)
            if alpha is not None:
                alpha = alpha.to(self.device)

        self.S = S  # Sub-intensity matrix
        if S is None:
            assert n is not None
            max_init_value = 1.0
            self.S = torch.tril(torch.rand(
                (n, n), device=self.device) * max_init_value)
            # Ensure diagonal elements are negative
            self.S.diagonal().copy_(torch.rand(
                n, device=self.device) * -max_init_value - 0.1)
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
        if w.device != self.device:
            self.to(w.device)

        w_flat = w.flatten()
        log_probs = []
        epsilon = 1e-10

        for wi in w_flat:
            try:
                scaled_S = torch.clamp(self.S * wi, min=-50.0, max=50.0)
                exp_Sw = torch.matrix_exp(scaled_S)
                density = self.alpha @ exp_Sw @ self.s

                if density <= 0:
                    density = torch.tensor(epsilon, device=w.device)

                log_probs.append(torch.log(density))
            except Exception as e:
                print(f"Error in log_prior calculation: {e}")
                # Return a very low probability instead of crashing
                log_probs.append(torch.tensor(-1e10, device=w.device))

        log_probs = torch.stack(log_probs)
        return log_probs.sum()

    def fit_params_to_data(self, data: torch.Tensor):
        if data.device != self.device:
            data = data.to(self.device)

        if self.device != 'cpu':
            torch.cuda.set_device(self.device)

        if self.device != 'cpu' and torch.cuda.is_available():
            if torch.cuda.get_device_capability(self.device.index if hasattr(self.device, 'index') else 0)[0] >= 7:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("CUDA optimization: TF32 precision enabled for matrix operations")

        orig_S = self.S.clone()
        orig_alpha = self.alpha.clone()

        print(f"PhaseType fitting - Before:")
        print(f"S matrix (first few values): {orig_S.flatten()[:5]}")
        print(f"alpha vector: {orig_alpha}")

        data = torch.abs(data.flatten())
        data = data[data > 0]

        if data.numel() == 0:
            raise ValueError(
                "No positive values in data after filtering zeros")

        data_mean = data.mean()
        data_std = data.std()

        outlier_threshold = data_mean + 3 * data_std
        outliers = data > outlier_threshold
        if outliers.any():
            num_outliers = outliers.sum().item()
            print(
                f"Detected {num_outliers} outliers in the data (>{outlier_threshold.item():.4f})")
            print(f"Clipping outliers to {outlier_threshold.item():.4f}")
            data = torch.clamp(data, max=outlier_threshold.item())

        if data_mean > 10.0 or data_std > 10.0:
            print(
                f"Normalizing data with mean={data_mean.item():.4f}, std={data_std.item():.4f}")
            data = (data - data_mean) / data_std + \
                1.0  # Shift to ensure positive values

        n = self.S.shape[0]  # Number of phases

        if not hasattr(self, 'S') or not hasattr(self, 'alpha'):
            self.S = torch.tril(torch.randn(n, n, device=device))
            self.alpha = torch.softmax(torch.randn(n, device=device), dim=0)
            self.s = -self.S.sum(dim=1)

        data, _ = torch.sort(data)

        print(
            f"Data stats: mean={data.mean().item():.4f}, std={data.std().item():.4f}, min={data.min().item():.4f}, max={data.max().item():.4f}")
        print(
            f"Fitting phase-type distribution with {n} phases to {len(data)} data points")

        # EM algorithm for phase-type distribution fitting
        max_iter = 20
        tolerance = 1e-4
        prev_log_likelihood = -float('inf')

        I = torch.eye(n, device=self.device)
        reg_eye = torch.eye(n, device=self.device) * 1e-6

        if self.device != 'cpu' and torch.cuda.is_available():
            try:
                free_memory = torch.cuda.get_device_properties(
                    self.device).total_memory - torch.cuda.memory_allocated(self.device)

                memory_per_point = n * n * 4 * 4  # 4 bytes per float32, 4 matrices per point
                optimal_batch_size = max(
                    1, min(100, int(free_memory * 0.3 / memory_per_point)))
                batch_size = optimal_batch_size
                print(
                    f"CUDA optimization: Using batch size {batch_size} based on available GPU memory")
            except Exception as e:
                print(f"Could not optimize batch size: {e}")
                batch_size = min(100, len(data))
        else:
            batch_size = min(100, len(data))

        if self.device != 'cpu':
            exp_Sx_buffer = torch.zeros((batch_size, n, n), device=self.device)
            E_i_buffer = torch.zeros((batch_size, n), device=self.device)

        for iteration in range(max_iter):
            # E-step: Compute expected sufficient statistics
            B = torch.zeros((n, n), device=self.device)
            z = torch.zeros(n, device=self.device)
            print(f"E-step: {iteration}")

            log_likelihood = 0.0

            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]

                exp_Sx_batch = []

                if self.device != 'cpu' and hasattr(torch, 'vmap') and len(batch) > 1:
                    try:
                        scaled_S_batch = torch.stack(
                            [torch.clamp(self.S * x, min=-50.0, max=50.0) for x in batch])

                        exp_Sx_batch = torch.vmap(
                            torch.matrix_exp)(scaled_S_batch)
                        print(
                            f"CUDA optimization: Used vectorized matrix exponential for batch of {len(batch)}")
                    except Exception as e:
                        print(
                            f"Vectorized matrix exponential failed: {e}, falling back to loop")
                        exp_Sx_batch = []
                        for x in batch:
                            try:
                                scaled_S = torch.clamp(
                                    self.S * x, min=-50.0, max=50.0)
                                exp_Sx = torch.matrix_exp(scaled_S)
                                exp_Sx_batch.append(exp_Sx)
                            except Exception as e:
                                print(
                                    f"Error in matrix exponential calculation: {e}")
                                scaled_S = torch.clamp(
                                    self.S * x, min=-20.0, max=20.0)
                                exp_Sx = torch.matrix_exp(scaled_S)
                                exp_Sx_batch.append(exp_Sx)

                        exp_Sx_batch = torch.stack(exp_Sx_batch)
                else:
                    for x in batch:
                        try:
                            scaled_S = torch.clamp(
                                self.S * x, min=-50.0, max=50.0)
                            exp_Sx = torch.matrix_exp(scaled_S)
                            exp_Sx_batch.append(exp_Sx)
                        except Exception as e:
                            print(
                                f"Error in matrix exponential calculation: {e}")
                            scaled_S = torch.clamp(
                                self.S * x, min=-20.0, max=20.0)
                            exp_Sx = torch.matrix_exp(scaled_S)
                            exp_Sx_batch.append(exp_Sx)

                    exp_Sx_batch = torch.stack(exp_Sx_batch)

                print(f"exp_Sx_batch: {exp_Sx_batch.shape}")

                if self.device != 'cpu':
                    try:
                        alpha_batch = self.alpha.unsqueeze(0).expand(
                            len(batch), -1)  # [batch_size, n]
                        s_batch = self.s.unsqueeze(0).expand(
                            len(batch), -1)  # [batch_size, n]

                        # First multiply alpha with each exp_Sx: [batch_size, 1, n] @ [batch_size, n, n] -> [batch_size, 1, n]
                        temp = torch.bmm(
                            alpha_batch.unsqueeze(1), exp_Sx_batch)

                        # Then multiply by s: [batch_size, 1, n] @ [batch_size, n, 1] -> [batch_size, 1, 1]
                        densities = torch.bmm(
                            temp, s_batch.unsqueeze(2)).squeeze(2).squeeze(1)
                        print(
                            f"CUDA optimization: Used batched matrix multiplication for densities")
                    except Exception as e:
                        print(
                            f"Batched density calculation failed: {e}, falling back to loop")
                        densities = torch.stack(
                            [self.alpha @ exp_Sx @ self.s for exp_Sx in exp_Sx_batch])
                else:
                    densities = torch.stack(
                        [self.alpha @ exp_Sx @ self.s for exp_Sx in exp_Sx_batch])

                epsilon = 1e-10
                if (densities <= 0).any():
                    print(
                        f"Warning: Found {(densities <= 0).sum().item()} non-positive density values")
                    densities = torch.clamp(densities, min=epsilon)

                log_likelihood += torch.log(densities).sum()

                # conditional expectations for the batch
                reg_matrix = -self.S + reg_eye

                if self.device != 'cpu' and hasattr(torch, 'linalg') and hasattr(torch.linalg, 'solve'):
                    try:
                        # Prepare the right-hand side for all batch elements
                        I_minus_exp_Sx = torch.stack(
                            [I - exp_Sx for exp_Sx in exp_Sx_batch])

                        reg_matrix_batch = reg_matrix.unsqueeze(
                            0).expand(len(batch), -1, -1)

                        solutions = torch.linalg.solve(
                            reg_matrix_batch, I_minus_exp_Sx)

                        E_i_batch = torch.matmul(self.alpha.unsqueeze(
                            0).unsqueeze(0), solutions).squeeze(1)
                        print(
                            f"CUDA optimization: Used batched linear algebra operations")
                    except Exception as e:
                        print(
                            f"Batched linear algebra failed: {e}, falling back to loop")
                        try:
                            E_i_batch = torch.stack([
                                self.alpha @ torch.linalg.solve(
                                    reg_matrix, I - exp_Sx)
                                for exp_Sx in exp_Sx_batch
                            ])
                        except torch._C._LinAlgError:
                            reg_matrix_pinv = torch.linalg.pinv(reg_matrix)
                            E_i_batch = torch.stack([
                                self.alpha @ reg_matrix_pinv @ (I - exp_Sx)
                                for exp_Sx in exp_Sx_batch
                            ])
                else:
                    try:
                        E_i_batch = torch.stack([
                            self.alpha @ torch.linalg.solve(
                                reg_matrix, I - exp_Sx)
                            for exp_Sx in exp_Sx_batch
                        ])
                    except torch._C._LinAlgError:
                        reg_matrix_pinv = torch.linalg.pinv(reg_matrix)
                        E_i_batch = torch.stack([
                            self.alpha @ reg_matrix_pinv @ (I - exp_Sx)
                            for exp_Sx in exp_Sx_batch
                        ])

                # Update sufficient statistics
                for idx, (x, exp_Sx, density) in enumerate(zip(batch, exp_Sx_batch, densities)):
                    E_i = E_i_batch[idx]

                    mask = (torch.eye(n, device=self.device)
                            == 0) & (self.S != 0)
                    if mask.any():
                        num_time_points = 30 if self.device != 'cpu' else 50
                        t_points = torch.linspace(
                            0, x.item(), num_time_points, device=self.device)

                        if self.device != 'cpu' and hasattr(torch, 'vmap'):
                            try:
                                scaled_S_time = torch.stack(
                                    [self.S * t for t in t_points])
                                # Clip values for numerical stability
                                scaled_S_time = torch.clamp(
                                    scaled_S_time, min=-50.0, max=50.0)

                                exp_St = torch.vmap(
                                    torch.matrix_exp)(scaled_S_time)
                                print(
                                    f"CUDA optimization: Used vectorized matrix exponential for integration")
                            except Exception as e:
                                print(
                                    f"Vectorized integration failed: {e}, falling back to loop")
                                exp_St = []
                                for t in t_points:
                                    try:
                                        scaled_S = torch.clamp(
                                            self.S * t, min=-50.0, max=50.0)
                                        exp_S = torch.matrix_exp(scaled_S)
                                        exp_St.append(exp_S)
                                    except Exception as e:
                                        print(
                                            f"Error in matrix exponential calculation for integration: {e}")
                                        scaled_S = torch.clamp(
                                            self.S * t, min=-20.0, max=20.0)
                                        exp_S = torch.matrix_exp(scaled_S)
                                        exp_St.append(exp_S)

                                exp_St = torch.stack(exp_St)
                        else:
                            exp_St = []
                            for t in t_points:
                                try:
                                    scaled_S = torch.clamp(
                                        self.S * t, min=-50.0, max=50.0)
                                    exp_S = torch.matrix_exp(scaled_S)
                                    exp_St.append(exp_S)
                                except Exception as e:
                                    print(
                                        f"Error in matrix exponential calculation for integration: {e}")
                                    scaled_S = torch.clamp(
                                        self.S * t, min=-20.0, max=20.0)
                                    exp_S = torch.matrix_exp(scaled_S)
                                    exp_St.append(exp_S)

                            exp_St = torch.stack(exp_St)

                        # For each non-zero transition
                        indices = mask.nonzero(as_tuple=True)
                        for idx in range(len(indices[0])):
                            i, j = indices[0][idx].item(
                            ), indices[1][idx].item()
                            exp_values = exp_St[:, i, j]
                            B_increment = self.alpha[i] * self.S[i,
                                                                 j] * torch.trapz(exp_values, t_points)

                            if torch.isnan(B_increment) or torch.isinf(B_increment):
                                print(
                                    f"Warning: Numerical issues in integration for B[{i},{j}]")
                                continue

                            max_increment = 100.0
                            if torch.abs(B_increment) > max_increment:
                                print(
                                    f"Warning: Large B increment: {B_increment.item():.4f}, clipping to {max_increment}")
                                B_increment = torch.clamp(
                                    B_increment, min=-max_increment, max=max_increment)

                            B[i, j] += B_increment

                    # Update z
                    if density < epsilon:
                        print(
                            f"Warning: Very small density value: {density.item():.10e}")
                        density = torch.tensor(epsilon, device=self.device)
                    z += self.alpha * exp_Sx @ self.s / density

            print(
                f'Iteration {iteration}, log_likelihood: {log_likelihood.item():.4f}')

            print(
                f"S matrix stats - min: {self.S.min().item():.4f}, max: {self.S.max().item():.4f}")
            print(
                f"alpha stats - min: {self.alpha.min().item():.4f}, max: {self.alpha.max().item():.4f}")
            print(
                f"s vector stats - min: {self.s.min().item():.4f}, max: {self.s.max().item():.4f}")

            if torch.isnan(log_likelihood) or torch.isinf(log_likelihood):
                print("Warning: Numerical issues detected in log likelihood")
                print(f"S contains NaN: {torch.isnan(self.S).any().item()}")
                print(
                    f"alpha contains NaN: {torch.isnan(self.alpha).any().item()}")
                print(f"s contains NaN: {torch.isnan(self.s).any().item()}")

                print("Stopping early due to numerical instability")
                if iteration > 0:
                    print("Restoring parameters from previous iteration")
                    max_abs_value = 10.0
                    self.S = torch.tril(torch.clamp(
                        self.S, min=-max_abs_value, max=max_abs_value))
                    diag_mask = self.S.diagonal() >= 0
                    if diag_mask.any():
                        self.S.diagonal()[diag_mask] = -0.1
                    self.s = -self.S.sum(dim=1)
                break

            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                break
            prev_log_likelihood = log_likelihood

            # M-step: Update parameters
            self.alpha = z / len(data)
            self.alpha = self.alpha / self.alpha.sum()  # Normalize

            for i in range(n):
                total_time_i = torch.sum(E_i_batch[:, i])
                if total_time_i > 0:
                    mask = torch.arange(n, device=self.device) != i
                    if total_time_i < epsilon:
                        print(
                            f"Warning: Very small total_time_i value: {total_time_i.item():.10e}")
                        total_time_i = torch.tensor(
                            epsilon, device=self.device)
                    self.S[i, mask] = B[i, mask] / total_time_i

            row_sums = torch.sum(self.S, dim=1)
            self.S.diagonal().copy_(-row_sums + self.S.diagonal())

            diag_mask = self.S.diagonal() >= 0
            if diag_mask.any():
                self.S.diagonal()[diag_mask] = -0.1

            self.s = -self.S.sum(dim=1)

            # limit the growth of S matrix values
            max_abs_value = 10.0  # Maximum absolute value allowed for S matrix elements
            self.S = torch.clamp(self.S, min=-max_abs_value, max=max_abs_value)

            self.s = -self.S.sum(dim=1)

            if self.device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()

                if iteration % 5 == 0:  # Only print every 5 iterations to reduce output
                    allocated = torch.cuda.memory_allocated(
                        self.device) / 1024**2
                    reserved = torch.cuda.memory_reserved(
                        self.device) / 1024**2
                    print(
                        f"CUDA memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

            if torch.isnan(self.S).any() or torch.isinf(self.S).any():
                print("Warning: Numerical issues detected in S matrix. Stopping early.")
                self.S = torch.tril(torch.clamp(
                    self.S, min=-max_abs_value, max=max_abs_value))
                diag_mask = self.S.diagonal() >= 0
                if diag_mask.any():
                    self.S.diagonal()[diag_mask] = -0.1
                self.s = -self.S.sum(dim=1)
                break

        # Ensure numerical stability
        self.S = torch.tril(self.S)
        diag_mask = self.S.diagonal() >= 0
        if diag_mask.any():
            self.S.diagonal()[diag_mask] = -0.1

        self.s = -self.S.sum(dim=1)

        # Normalize alpha
        self.alpha = self.alpha / self.alpha.sum()

        print(f"PhaseType fitting - After:")
        print(f"S matrix (first few values): {self.S.flatten()[:5]}")
        print(f"alpha vector: {self.alpha}")

        try:
            fitted_mean = self.alpha @ torch.linalg.inv(-self.S) @ torch.ones(
                n, device=self.device)
            print(
                f"Fitted distribution mean: {fitted_mean.item():.4f} (Data mean: {data.mean().item():.4f})")
        except:
            print("Could not calculate fitted distribution mean")

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return self

    def get_param_device(self):
        """Get the device of the distribution parameters"""
        return self.S.device

    def to(self, device):
        """Move all tensors to the specified device"""
        if hasattr(self, 'S') and self.S is not None:
            self.S = self.S.to(device)

        if hasattr(self, 'alpha') and self.alpha is not None:
            self.alpha = self.alpha.to(device)

        if hasattr(self, 's') and self.s is not None:
            self.s = self.s.to(device)

        self.device = device

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
