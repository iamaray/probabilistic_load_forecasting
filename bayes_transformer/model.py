import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import taskbalance
from .utils import rmse_loss
from .layers import BayesianLinear
from .utils import variational_estimator
from utils import model_saver

from preprocessing import MinMaxNorm, StandardScaleNorm
################################################################################################################


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
###########################


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h

        self.h = h
        self.linears = clones(BayesianLinear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
####################


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
###############################


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
#####################


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
############################


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = BayesianLinear(d_model, d_ff)
        self.w_2 = BayesianLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
#####################


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout, d_ff=128, N=2, h=8):
        super(EncoderLayer, self).__init__()
        self.size = d_model
        self.d_ff = d_ff
        self.N = N
        self.h = h
        self.self_attn = MultiHeadedAttention(
            h=h, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.sublayer = clones(SublayerConnection(
            size=d_model, dropout=dropout), N=N)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)
###############################


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, dropout=0.1, d_ff=128, N=2, h=8):
        super(Encoder, self).__init__()
        layer = EncoderLayer(d_model=d_model, dropout=dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
###############################


class DecoderLayer(nn.Module):
    def __init__(self, d_model, dropout, h=8, d_ff=128, N=3):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(
            h=h, d_model=d_model, dropout=dropout)
        self.src_attn = MultiHeadedAttention(
            h=h, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.sublayer = clones(SublayerConnection(
            size=d_model, dropout=dropout), N=N)

    def forward(self, memory, x):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)
#########################


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_targets, num_aux_feats, window_length, ahead, dropout, h, N, d_ff):
        super(Decoder, self).__init__()
        feats = num_targets + num_aux_feats
        layer = DecoderLayer(
            d_model=d_model, dropout=dropout, h=h, d_ff=d_ff, N=N)
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)
        self.outputLinear = BayesianLinear(d_model, feats)
        self.outputLinear1 = BayesianLinear(window_length*feats, window_length)
        self.outputLinear2 = BayesianLinear(window_length, ahead)

    def forward(self, memory, decoderINPUT):
        for layer in self.layers:
            memory = layer(memory, decoderINPUT)
        return self.outputLinear2(self.outputLinear1(torch.flatten(input=self.outputLinear(self.norm(memory)), start_dim=1)))
#################


@variational_estimator
class BayesianMDeT(nn.Module):
    """
    BayesianDeT
    """

    def __init__(
            self,
            ahead,
            num_targets,
            num_aux_feats,
            window_len,
            name="BSMDeT",
            d_model=32,
            encoder_layers=2,
            encoder_d_ff=128,
            encoder_sublayers=2,
            encoder_h=8,
            encoder_dropout=0.1,
            decoder_layers=2,
            decoder_dropout=0.1,
            decoder_h=8,
            decoder_d_ff=128,
            decoder_sublayers=3):

        super(BayesianMDeT, self).__init__()
        self.name = name
        self.num_targets = num_targets
        self.num_aux_feats = num_aux_feats
        self.window_len = window_len
        self.d_model = d_model
        num_feats = num_targets + num_aux_feats
        self.encoder = Encoder(num_layers=encoder_layers, d_model=d_model,
                               dropout=encoder_dropout, d_ff=encoder_d_ff, N=encoder_sublayers, h=encoder_h)

        self.decoders = nn.ModuleList()
        for _ in range(num_targets):
            self.decoders.append(Decoder(num_layers=decoder_layers, d_model=d_model, ahead=ahead, num_targets=num_targets,
                                         num_aux_feats=num_aux_feats, window_length=window_len, dropout=decoder_dropout, h=decoder_h, N=decoder_sublayers, d_ff=decoder_d_ff))

        self.encoderLinear = BayesianLinear(num_feats, d_model)
        self.decoder_linear_layers = nn.ModuleList()
        for _ in range(num_targets):
            self.decoder_linear_layers.append(
                BayesianLinear(1 + num_aux_feats, d_model))

    def forward(self, input: torch.Tensor):
        # x = input.transpose(1, 2)
        x = input
        # print(x.shape)
        # x: input data. shape (batch, window_len, num_targets + num_aux_feats)
        encoder_output = self.encoder(self.encoderLinear(x))
        # aux: Auxiliary information. shape (batch, window_len, num_aux_feats)
        aux = x[:, :, self.num_targets:]

        inputs = [
            torch.cat([x[:, :, i:i+1], aux], dim=2)
            for i in range(self.num_targets)
        ]

        outputs = [self.decoders[i](encoder_output, self.decoder_linear_layers[i](
            inputs[i])) for i in range(self.num_targets)]
        outputs = torch.cat(outputs, dim=-1)

        if len(outputs.shape) == 2:
            outputs = outputs.unsqueeze(-1)
        return outputs

###########################################


class BSMDeTWrapper(nn.Module):
    def __init__(self,
                 ahead=24,
                 num_targets=1,
                 num_aux_feats=0,
                 window_len=168,
                 cuda=False,
                 name="BSMDeT",
                 d_model=32,
                 encoder_layers=2,
                 encoder_d_ff=128,
                 encoder_sublayers=2,
                 encoder_h=8,
                 encoder_dropout=0.1,
                 decoder_layers=2,
                 decoder_dropout=0.1,
                 decoder_h=8,
                 decoder_d_ff=128,
                 decoder_sublayers=3,
                 lr=0.001):
        super(BSMDeTWrapper, self).__init__()
        self.num_targets = num_targets
        self.ahead = ahead
        self.num_aux_feats = num_aux_feats
        self.window_len = window_len
        self.cuda = cuda
        self.name = name
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_d_ff = encoder_d_ff
        self.encoder_sublayers = encoder_sublayers
        self.encoder_h = encoder_h
        self.encoder_dropout = encoder_dropout
        self.decoder_layers = decoder_layers
        self.decoder_dropout = decoder_dropout
        self.decoder_h = decoder_h
        self.decoder_d_ff = decoder_d_ff
        self.decoder_sublayers = decoder_sublayers

        self.create(lr=lr)
        self.cuda = cuda

        self.scaler = MinMaxNorm()
    ###########################

    def create(self, lr=0.001):
        self.model = BayesianMDeT(
            ahead=self.ahead,
            num_targets=self.num_targets,
            num_aux_feats=self.num_aux_feats,
            window_len=self.window_len,
            name=self.name,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            encoder_d_ff=self.encoder_d_ff,
            encoder_sublayers=self.encoder_sublayers,
            encoder_h=self.encoder_h,
            encoder_dropout=self.encoder_dropout,
            decoder_layers=self.decoder_layers,
            decoder_dropout=self.decoder_dropout,
            decoder_h=self.decoder_h,
            decoder_d_ff=self.decoder_d_ff,
            decoder_sublayers=self.decoder_sublayers)

        self.BayesianWeightLinear = taskbalance(num=self.num_targets)

        if self.cuda:
            self.model.cuda()
            self.BayesianWeightLinear.cuda()

        print('    Total params: %.2fM' %
              (self.get_nb_parameters() / 1000000.0))
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                           {'params': self.BayesianWeightLinear.parameters()}],
                                          lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
    ###########################

    def fit(self, in_x, in_y, samples=1):
        x, y = in_x.transpose(1, 2), in_y.transpose(1, 2)
        x_scaled = self.scaler.fit_transform(x)
        # x, y = in_x, in_y
        self.optimizer.zero_grad()
        # print(x.shape, y.shape)
        ave_losses, kl = self.model.sample_elbo_m(inputs=x_scaled,
                                                  labels=y,
                                                  num_targets=self.num_targets,
                                                  sample_nbr=samples,
                                                  scaler=self.scaler)

        overall_loss = self.BayesianWeightLinear(ave_losses)
        overall_loss = overall_loss + kl

        p_mu = self.BayesianWeightLinear.weight_mu
        p_rho = self.BayesianWeightLinear.weight_rho

        overall_loss.backward()
        self.optimizer.step()
        print('here0', ave_losses.shape)
        return overall_loss, ave_losses, p_mu, p_rho
    ###########################

    def test(self, in_test, samples=10):
        x_test = in_test.transpose(1, 2)
        batch_size = x_test.shape[0]
        # outputs = [np.zeros((0, batch_size, self.ahead))
        #            for _ in range(self.num_targets)]

        # for _ in range(samples):
        #     model_outputs = self.model(x_test)

        # for i, output in enumerate(model_outputs):
        #     # output_np = torch.permute(output.cpu().detach(),
        #     #                           (-1, output.shape[0], output.shape[1]))
        #     # outputs[i] = torch.cat(
        #     #     [outputs[i], output_np], axis=0).unsqueeze(-1)
        #     outputs[i] = torch.cat(
        #         [outputs[i]], axis=0).unsqueeze(-1)

        return torch.stack(
            [self.model(x_test).cpu().detach()
             for _ in range(samples)],
            dim=-1)
    ###########################

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())
#############
