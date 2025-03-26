import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from typing import Optional

# Variational Dropout remains unchanged.


class VariationalDropout(nn.Module):
    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(
                2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(
                2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

# A custom LSTM that applies variational dropout on inputs, weights, and outputs.


class LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float = 0., dropoutw: float = 0., dropouto: float = 0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti, batch_first=batch_first)
        self.output_drop = VariationalDropout(
            dropouto, batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers.
        Bias is 0 except for forget gate.
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                param.data = F.dropout(
                    param.data, p=self.dropoutw, training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state

# Updated DeepAR model that now incorporates an embedding layer.


class DeepAR(nn.Module):
    def __init__(self,
                 num_class: int = 100,
                 embedding_dim: int = 32,
                 covariate_size: int = 4,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 predict_steps: int = 24,
                 predict_start: int = 168):
        """
        Args:
          num_class: Number of distinct time series identifiers.
          embedding_dim: Dimensionality of the embedding vectors.
          covariate_size: Number of exogenous features per time step.
          hidden_size: Number of hidden units in the LSTM.
          num_layers: Number of LSTM layers.
          dropout: Dropout probability applied via variational dropout.
        """
        super(DeepAR, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.covariate_size = covariate_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_steps = predict_steps
        self.predict_start = predict_start
        # input len: previous target (1) + covariates + embedding.
        self.input_size = 1 + covariate_size + embedding_dim
        self.embedding = nn.Embedding(num_class, embedding_dim)

        # self.lstm = LSTM(self.input_size, hidden_size,
        self.lstm = nn.LSTM(input_size=self.input_size,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bias=True,
                            batch_first=False,
                            dropout=dropout)

        # init LSTM forget gate bias to 1.0
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(
            in_features=self.hidden_size * self.num_layers,
            out_features=1)

        self.distribution_presigma = nn.Linear(
            in_features=self.hidden_size * self.num_layers,
            out_features=1)

        self.distribution_sigma = nn.Softplus()

    def forward(self, target, covariates, idx, h, c):
        """
        Args:
          target: Tensor of shape (1, batch) containing target values.
          covariates: Tensor of shape (1, batch, covariate_size) containing covariate features.
          idx: Tensor of shape (1, batch); a single integer denoting which time series the input corresponds to.

        Returns:
          loss: Scalar tensor, the average negative log likelihood computed over observed steps.
          predictions: A list of tuples (mean, sigma) for each time step.
        """
        batch_size, seq_len = target.size()
        device = target.device

        embed = self.embedding(idx)  # shape: (1, batch, embedding_dim)
        # shape: (1, batch, covariate_size + 1)
        x_cat = torch.cat([target.unsqueeze(-1), covariates], dim=-1)
        # shape: (1, batch, input_size)
        lstm_in = torch.cat([x_cat, embed], dim=-1)
        out, (h, c) = self.lstm(lstm_in, (h, c))

        # out = out.squeeze(0)  # shape: (batch, hidden_size)
        # mu = self.distribution_mu(out)  # shape: (batch, 1)
        # presigma = self.distribution_presigma(out)  # shape: (batch, 1)
        # sigma = self.distribution_sigma(presigma)  # shape: (batch, 1)

        h_perm = h.permute(1, 2, 0).contiguous().view(h.shape[1], -1)
        presigma = self.distribution_presigma(h_perm)
        mu = self.distribution_mu(h_perm)
        sigma = self.distribution_sigma(presigma)

        return mu.squeeze(-1), sigma.squeeze(-1), h, c

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=self.embedding.weight.device)

    def init_cell(self, batch_size):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=self.embedding.weight.device)

    def forecast(
            self,
            test_loader=None,
            label=None,
            covariates=None,
            num_samples=100,
            idx=None,
            data_norm=None):
        """
        Generate probabilistic forecasts for future time steps using a test data loader.

        Args:
            test_loader: DataLoader containing test data batches.
            label: Optional tensor of shape (batch, seq_len) containing target values.
            covariates: Optional tensor of shape (batch, seq_len, covariate_size) containing covariate features.
            num_samples: Number of Monte Carlo samples to draw for each forecast.
            idx: Optional tensor of shape (batch,) containing time series identifiers.
            data_norm: Optional data normalization transform.

        Returns:
            all_samples: Tensor containing sampled trajectories for each batch.
            all_means: Tensor containing the mean forecasts for each batch.
            all_sigmas: Tensor containing the standard deviations for each batch.
        """
        self.eval()

        if test_loader is not None:
            # Get the first batch from test loader
            batch = next(iter(test_loader))
            label = batch[0]  # target values
            covariates = batch[1]  # covariates
            mask = batch[2]  # mask showing where prediction starts

            # Determine forecast start index from mask
            # Mask indicates observed values
            # Use first sequence in batch
            forecast_start = mask[0].sum().item()
        else:
            # If no test_loader is provided, use the provided tensors
            forecast_start = self.predict_start
            if mask is None:
                # Assume prediction starts at predict_start if not specified
                mask = torch.ones_like(label)
                mask[:, self.predict_start:] = 0

        batch_size = label.size(0)
        transformed_label = label
        transformed_covariates = covariates
        print("cov shape test", transformed_covariates.shape)
        # Create default time series indices if not provided
        if idx is None:
            idx = torch.zeros(batch_size, dtype=torch.long,
                              device=label.device)

        # Apply data normalization if provided
        if data_norm is not None:
            data_norm.set_device(label.device)
            transform_in = torch.cat(
                [label.unsqueeze(-1), covariates], dim=-1)
            transformed = data_norm.transform(transform_in)

            transformed_label = transformed[:, :, 0]
            transformed_covariates = transformed[:, :, 1:]

        device = transformed_label.device
        predict_steps = transformed_label.size(1) - forecast_start

        with torch.no_grad():
            # Initialize hidden and cell states
            h = self.init_hidden(batch_size)
            c = self.init_cell(batch_size)

            # Pre-condition the model on observed data
            for t in range(forecast_start):
                current_target = transformed_label[:, t].unsqueeze(0)
                current_covariates = transformed_covariates[:, t, :].unsqueeze(
                    0)
                # Forward pass to update hidden state
                _, _, h, c = self(
                    target=current_target,
                    covariates=current_covariates,
                    idx=idx.unsqueeze(0),
                    h=h,
                    c=c
                )

            # Generate samples from the predictive distribution
            samples = torch.zeros(num_samples, batch_size,
                                  predict_steps, 1, device=device)
            for j in range(num_samples):
                sample_h = h.clone()
                sample_c = c.clone()
                forecast_sequence = transformed_label.clone()

                for i in range(predict_steps):
                    t = forecast_start + i
                    current_target = forecast_sequence[:, t-1].unsqueeze(0)
                    current_covariates = transformed_covariates[:, t, :].unsqueeze(
                        0)

                    # Generate prediction
                    mu, sigma, sample_h, sample_c = self(
                        target=current_target,
                        covariates=current_covariates,
                        idx=idx.unsqueeze(0),
                        h=sample_h,
                        c=sample_c
                    )

                    # Sample from the predicted distribution
                    # noise = torch.randn_like(mu)
                    # sample = mu + sigma * noise

                    sample = torch.distributions.Normal(
                        loc=mu, scale=sigma).sample()

                    samples[j, :, i, :] = sample.unsqueeze(-1)

                    # Update the forecast for the next step
                    if i < predict_steps - 1:
                        forecast_sequence[:, t] = sample

            sample_means = samples.mean(dim=0)
            sample_sigmas = samples.std(dim=0)

            # Reverse the normalization if applicable
            if data_norm is not None:
                print('data norm shapes', data_norm.mean.shape,
                      data_norm.std.shape)
                print('data norm items', data_norm.mean, data_norm.std)
                data_norm.set_device(device)
                # samples = data_norm.reverse(samples)
                samples = samples[..., 0:1] * \
                    data_norm.std[..., 0:1] + data_norm.mean[..., 0:1]
                # sample_means = data_norm.reverse(
                #     sample_means.unsqueeze(-1))

                # sample_sigmas = data_norm.inverse_transform(
                #     sample_sigmas, dim=1, is_std=True)

        return samples, sample_means, sample_sigmas


def neg_gaussian_log_likelihood(mu, sigma, target):
    """Calculate the negative log likelihood of a Gaussian distribution."""
    labeled_idx = target != 0
    return 0.5 * torch.log(2 * math.pi * sigma[labeled_idx]**2) + (target[labeled_idx] - mu[labeled_idx])**2 / (2 * sigma[labeled_idx]**2)


def neg_cauchy_log_likelihood(mu, sigma, target):
    labeled_idx = target != 0
    return torch.log(1 + (target[labeled_idx] - mu[labeled_idx])**2 / (sigma[labeled_idx]**2))


def neg_student_t_log_likelihood(mu, sigma, target, df):
    labeled_idx = target != 0
    return torch.log(1 + (target[labeled_idx] - mu[labeled_idx])**2 / (df * sigma[labeled_idx]**2))


def neg_poisson_log_likelihood(mu, target):
    labeled_idx = target != 0
    return -mu[labeled_idx] + target[labeled_idx] * torch.log(mu[labeled_idx]) - torch.lgamma(target[labeled_idx] + 1)


def neg_negative_binomial_log_likelihood(mu, sigma, target):
    labeled_idx = target != 0
    return torch.log(torch.lgamma(target[labeled_idx] + sigma[labeled_idx]) - torch.lgamma(sigma[labeled_idx]) - torch.lgamma(target[labeled_idx] + 1)) + sigma[labeled_idx] * torch.log(sigma[labeled_idx]) + target[labeled_idx] * torch.log(mu[labeled_idx]) - (mu[labeled_idx] + sigma[labeled_idx]) * torch.log(mu[labeled_idx] + sigma[labeled_idx])


def neg_gamma_log_likelihood(mu, sigma, target):
    labeled_idx = target != 0
    return torch.log(torch.lgamma(target[labeled_idx] / sigma[labeled_idx]) - torch.lgamma(target[labeled_idx]) - target[labeled_idx] * torch.log(sigma[labeled_idx])) + target[labeled_idx] * (1 - torch.log(sigma[labeled_idx]))


def neg_inverse_gaussian_log_likelihood(mu, sigma, target):
    labeled_idx = target != 0
    return torch.log(torch.lgamma(target[labeled_idx] / sigma[labeled_idx]) - torch.lgamma(target[labeled_idx]) - target[labeled_idx] * torch.log(sigma[labeled_idx])) + target[labeled_idx] * (1 - torch.log(sigma[labeled_idx]))


def neg_log_normal_log_likelihood(mu, sigma, target):
    labeled_idx = target != 0
    return torch.log(torch.lgamma(target[labeled_idx] / sigma[labeled_idx]) - torch.lgamma(target[labeled_idx]) - target[labeled_idx] * torch.log(sigma[labeled_idx])) + target[labeled_idx] * (1 - torch.log(sigma[labeled_idx]))
