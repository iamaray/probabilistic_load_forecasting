from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from typing import *


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


class LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float = 0.,
                 dropoutw: float = 0., dropouto: float = 0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
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
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


class Net(nn.Module):
    def __init__(self, params):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(Net, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params.num_class, params.embedding_dim)

        self.lstm = nn.LSTM(input_size=1+params.cov_dim+params.embedding_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)
        '''self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)'''
        # initialize LSTM forget gate bias to be 1 as recommended by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_presigma = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x, idx, hidden, cell):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            idx ([1, batch_size]): one integer denoting the time series id
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        onehot_embed = self.embedding(
            idx)  # TODO: is it possible to do this only once per window instead of per step?
        lstm_input = torch.cat((x, onehot_embed), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(
            1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        # softplus to make sure standard deviation is positive
        sigma = self.distribution_sigma(pre_sigma)
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def test(self, x, v_batch, id_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.predict_steps,
                                  device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                         id_batch, decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(
                        mu_de, sigma_de)
                    pred = gaussian.sample()  # not scaled
                    samples[j, :, t] = pred * v_batch[:, 0] + v_batch[:, 1]
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred

            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(
                batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(
                batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                     id_batch, decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma


def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(
        mu[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)
