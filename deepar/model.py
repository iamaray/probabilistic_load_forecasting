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
                 dropout: float = 0.0):
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
        # The input now is composed of: previous target (1) + covariates + embedding.
        self.input_size = 1 + covariate_size + embedding_dim

        # Create the embedding layer for time series id.
        self.embedding = nn.Embedding(num_class, embedding_dim)

        # Use our custom LSTM that applies variational dropout.
        self.lstm = LSTM(self.input_size, hidden_size,
                         num_layers, dropouti=dropout, batch_first=False)

        # A linear layer to predict distribution parameters (mean and pre-sigma).
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, target, covariates, mask, idx: Optional[torch.Tensor] = None):
        """
        Args:
          target: Tensor of shape (batch, seq_len) containing target values.
          covariates: Tensor of shape (batch, seq_len, covariate_size) containing covariate features.
          mask: Boolean Tensor of shape (batch, seq_len) with True for observed data.
          idx: Optional tensor of shape (1, batch) containing time series ids.
               If not provided, defaults to a range [0, batch_size-1].

        Returns:
          loss: Scalar tensor, the average negative log likelihood computed over observed steps.
          predictions: A list of tuples (mean, sigma) for each time step.
        """
        batch_size, seq_len = target.size()
        device = target.device

        # If no ids are provided, assign each sample an id equal to its index.
        if idx is None:
            idx = torch.arange(batch_size, device=device).unsqueeze(
                0)  # shape: (1, batch)

        # Initialize LSTM hidden and cell states.
        h = torch.zeros(self.num_layers, batch_size,
                        self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size,
                        self.hidden_size, device=device)

        # Initial autoregressive input: zeros.
        input_prev = torch.zeros(batch_size, 1, device=device)

        total_loss = 0.0
        count = 0
        predictions = []

        # Unroll over the time sequence.
        for t in range(seq_len):
            # Extract covariates at time t.
            cov_t = covariates[:, t]  # shape: (batch, covariate_size)
            # Obtain the embedding for each sample.
            # Note: The same id is used for all time steps of a given sample.
            emb = self.embedding(idx)  # shape: (1, batch, embedding_dim)
            # Concatenate previous target, current covariates, and embedding.
            # input_prev: (batch, 1); cov_t: (batch, covariate_size); emb.squeeze(0): (batch, embedding_dim)
            # shape: (batch, 1+covariate_size+embedding_dim)
            lstm_in = torch.cat([input_prev, cov_t, emb.squeeze(0)], dim=1)
            lstm_in = lstm_in.unsqueeze(0)  # shape: (1, batch, input_size)

            # Pass through the LSTM.
            out, (h, c) = self.lstm(lstm_in, (h, c))
            out = out.squeeze(0)  # shape: (batch, hidden_size)

            # Predict distribution parameters.
            params = self.fc(out)  # shape: (batch, 2)
            mean = params[:, 0]    # predicted mean
            sigma = F.softplus(params[:, 1]) + 1e-6  # ensure positivity

            predictions.append((mean, sigma))

            # Compute negative log likelihood only for observed time steps.
            z_t = target[:, t]
            observed = mask[:, t]
            if observed.sum() > 0:
                z_obs = z_t[observed]
                mean_obs = mean[observed]
                sigma_obs = sigma[observed]
                nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma_obs) + \
                    0.5 * ((z_obs - mean_obs) / sigma_obs) ** 2
                total_loss += nll.mean()
                count += 1

            # For autoregressive prediction: use true value if observed, else sample.
            sample = mean + sigma * torch.randn_like(mean)
            next_input = torch.where(observed, z_t, sample)
            input_prev = next_input.unsqueeze(1)  # shape: (batch, 1)

        loss = total_loss / count if count > 0 else total_loss
        return loss, predictions

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.embedding.weight.device)

    def init_cell(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.embedding.weight.device)

    def forecast(self, test_loader, num_samples=100):
        """
        Generate probabilistic forecasts for future time steps using a test data loader.

        Args:
            test_loader: DataLoader containing test data with batches of (target, covariates, mask) for AR models
                         or (X, Y) for non-AR models.
            num_samples: Number of Monte Carlo samples to draw for each forecast.

        Returns:
            all_samples: List of tensors containing sampled trajectories for each batch.
            all_means: List of tensors containing the mean forecasts for each batch.
            all_sigmas: List of tensors containing the standard deviations for each batch.
        """
        self.eval()  # Set model to evaluation mode

        all_samples = []
        all_means = []
        all_sigmas = []

        with torch.no_grad():
            for batch in test_loader:
                # Check if we're using AR model (3 elements in batch) or non-AR model (2 elements)
                if len(batch) == 3:  # AR model: (target, covariates, mask)
                    context_target, context_covariates, mask = batch

                    # For AR models, we need to split the data into context and forecast parts
                    # based on the mask (True for context, False for forecast)
                    batch_size = context_target.size(0)
                    device = context_target.device

                    # Find where the forecast part begins (first False in mask)
                    # Assuming all samples have same context length
                    forecast_start = mask[0].sum().item()
                    context_length = forecast_start
                    forecast_length = context_target.size(1) - context_length

                    # Split the data
                    forecast_target = context_target[:, context_length:]
                    forecast_covariates = context_covariates[:,
                                                             context_length:]
                    context_target = context_target[:, :context_length]
                    context_covariates = context_covariates[:, :context_length]
                    mask = mask[:, :context_length]

                    # No specific time series IDs provided
                    idx = torch.arange(batch_size, device=device).unsqueeze(0)

                else:  # non-AR model: (X, Y)
                    X, Y = batch
                    # Assuming first column is the target
                    context_target = X[:, :, 0]
                    context_covariates = X[:, :, 1:]  # Rest are covariates
                    forecast_covariates = torch.zeros_like(
                        context_covariates[:, :Y.size(1), :])  # Placeholder
                    # Assume all observed
                    mask = torch.ones_like(context_target, dtype=torch.bool)
                    batch_size = context_target.size(0)
                    context_length = context_target.size(1)
                    forecast_length = Y.size(1)
                    device = context_target.device
                    idx = torch.arange(batch_size, device=device).unsqueeze(0)

                # Initialize storage for samples and distribution parameters
                samples = torch.zeros(num_samples, batch_size,
                                      forecast_length, device=device)
                means = torch.zeros(batch_size, forecast_length, device=device)
                sigmas = torch.zeros(
                    batch_size, forecast_length, device=device)

                # For each Monte Carlo sample
                for s in range(num_samples):
                    # Initialize LSTM hidden and cell states
                    h = self.init_hidden(batch_size).to(device)
                    c = self.init_cell(batch_size).to(device)

                    # Initial autoregressive input: zeros
                    input_prev = torch.zeros(batch_size, 1, device=device)

                    # First, run through the context window to condition the model
                    for t in range(context_length):
                        # Extract covariates at time t
                        # shape: (batch, covariate_size)
                        cov_t = context_covariates[:, t]

                        # Get embeddings for each time series
                        # shape: (1, batch, embedding_dim)
                        emb = self.embedding(idx)

                        # Concatenate previous target, current covariates, and embedding
                        lstm_in = torch.cat(
                            [input_prev, cov_t, emb.squeeze(0)], dim=1)
                        # shape: (1, batch, input_size)
                        lstm_in = lstm_in.unsqueeze(0)

                        # Pass through the LSTM
                        out, (h, c) = self.lstm(lstm_in, (h, c))
                        out = out.squeeze(0)  # shape: (batch, hidden_size)

                        # Predict distribution parameters
                        params = self.fc(out)  # shape: (batch, 2)
                        mean = params[:, 0]    # predicted mean
                        # ensure positivity
                        sigma = F.softplus(params[:, 1]) + 1e-6

                        # For autoregressive prediction: use true value if observed, else sample
                        z_t = context_target[:, t]
                        observed = mask[:, t]
                        sample = mean + sigma * torch.randn_like(mean)
                        next_input = torch.where(observed, z_t, sample)
                        input_prev = next_input.unsqueeze(
                            1)  # shape: (batch, 1)

                    # Now generate forecasts for future time steps
                    for t in range(forecast_length):
                        # Extract covariates at future time t
                        # shape: (batch, covariate_size)
                        if len(batch) == 3:  # AR model
                            cov_t = forecast_covariates[:, t]
                        else:  # non-AR model, use Y's time step for covariates if available
                            # This is a placeholder, in practice you might want to use actual future covariates
                            # Use last context covariate as placeholder
                            cov_t = context_covariates[:, -1]

                        # Get embeddings for each time series
                        # shape: (1, batch, embedding_dim)
                        emb = self.embedding(idx)

                        # Concatenate previous prediction, current covariates, and embedding
                        lstm_in = torch.cat(
                            [input_prev, cov_t, emb.squeeze(0)], dim=1)
                        # shape: (1, batch, input_size)
                        lstm_in = lstm_in.unsqueeze(0)

                        # Pass through the LSTM
                        out, (h, c) = self.lstm(lstm_in, (h, c))
                        out = out.squeeze(0)  # shape: (batch, hidden_size)

                        # Predict distribution parameters
                        params = self.fc(out)  # shape: (batch, 2)
                        mean = params[:, 0]    # predicted mean
                        # ensure positivity
                        sigma = F.softplus(params[:, 1]) + 1e-6

                        # Store the distribution parameters (only for the first sample)
                        if s == 0:
                            means[:, t] = mean
                            sigmas[:, t] = sigma

                        # Sample from the predicted distribution
                        sample = mean + sigma * torch.randn_like(mean)
                        samples[s, :, t] = sample

                        # Use the sample as the next input
                        input_prev = sample.unsqueeze(1)  # shape: (batch, 1)

                all_samples.append(samples)
                all_means.append(means)
                all_sigmas.append(sigmas)

        return all_samples, all_means, all_sigmas
