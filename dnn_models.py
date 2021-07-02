import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(
        torch.arange(
            x.size(1) - 1, -1, -1
        ), ('cpu', 'cuda')[x.is_cuda]
    )().long(), :]
    return x.view(xsize)


class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
        stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50,
        min_band_hz=50
    ):

        super(SincConv, self).__init__()

        if in_channels != 1:
            msg = 'SincConv only support one input channel ' + \
                  f'(here, in_channels = {in_channels})'
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetric)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filter banks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / self.kernel_size
        )

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(
            1, -1
        ) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz,
            self.sample_rate / 2
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Equivalent of Eq.4 of the reference paper
        # (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET).
        # I just have expanded the sinc and simplified the terms.
        # This way I avoid several useless computations.
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(
            f_times_t_low
        )) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size
        )

        return F.conv1d(
            waveforms, self.filters, stride=self.stride, padding=self.padding,
            dilation=self.dilation, bias=None, groups=1
        )


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(
            1
        )  # initialized like this, but not used in forward!


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()

        self.input_dim = int(options['input_dim'])
        self.fc_lay = options['fc_lay']
        self.fc_drop = options['fc_drop']
        self.fc_use_batch_norm = options['fc_use_batch_norm']
        self.fc_use_layer_norm = options['fc_use_layer_norm']
        self.fc_use_layer_norm_inp = options['fc_use_layer_norm_inp']
        self.fc_use_batch_norm_inp = options['fc_use_batch_norm_inp']
        self.fc_act = options['fc_act']

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.fc_use_layer_norm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization    
        if self.fc_use_batch_norm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        self.N_fc_lay = len(self.fc_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_fc_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.fc_drop[i]))

            # activation
            self.act.append(act_fun(self.fc_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.fc_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.fc_lay[i], momentum=0.05))

            if self.fc_use_layer_norm[i] or self.fc_use_batch_norm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(
                nn.Linear(current_input, self.fc_lay[i], bias=add_bias)
            )

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.fc_lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.fc_lay[i])),
                    np.sqrt(0.01 / (current_input + self.fc_lay[i]))
                )
            )
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))

            current_input = self.fc_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.fc_use_layer_norm_inp):
            x = self.ln0((x))

        if bool(self.fc_use_batch_norm_inp):
            x = self.bn0((x))

        for i in range(self.N_fc_lay):

            if self.fc_act[i] != 'linear':

                if self.fc_use_layer_norm[i]:
                    x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

                if self.fc_use_batch_norm[i]:
                    x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

                if self.fc_use_batch_norm[i] is False and self.fc_use_layer_norm[
                    i] is False:
                    x = self.drop[i](self.act[i](self.wx[i](x)))

            else:
                if self.fc_use_layer_norm[i]:
                    x = self.drop[i](self.ln[i](self.wx[i](x)))

                if self.fc_use_batch_norm[i]:
                    x = self.drop[i](self.bn[i](self.wx[i](x)))

                if self.fc_use_batch_norm[i] is False and self.fc_use_layer_norm[
                    i] is False:
                    x = self.drop[i](self.wx[i](x))

        return x


class SincNet(nn.Module):

    def __init__(
        self,
        # number of samples, the default is 200ms worth of samples at 16kHz
        input_dim=3200,

        # sampling frequency
        fs=16000,

        # number of filters for each layer
        num_filters=[80, 60, 60],

        # size of filter for each layer
        filter_sizes=[251, 5, 5],

        # size of the max pooling kernel
        max_pool_sizes=[3, 3, 3],

        cnn_use_layer_norm_inp=True,
        cnn_use_batch_norm_inp=False,
        cnn_use_layer_norm=[True, True, True],
        cnn_use_batch_norm=[False, False, False],

        cnn_activations=['leaky_relu', 'leaky_relu', 'leaky_relu'],
        cnn_drops=[0.0, 0.0, 0.0]

    ):
        """
        conv -> max pool -> [layer_norm | batch_norm] -> activation -> drop
        """
        super(SincNet, self).__init__()

        self.cnn_N_filter = num_filters
        self.cnn_len_filter = filter_sizes
        self.cnn_max_pool_len = max_pool_sizes

        self.cnn_act = cnn_activations
        self.cnn_drop = cnn_drops

        self.cnn_use_layer_norm = cnn_use_layer_norm
        self.cnn_use_batch_norm = cnn_use_batch_norm
        self.cnn_use_layer_norm_inp = cnn_use_layer_norm_inp
        self.cnn_use_batch_norm_inp = cnn_use_batch_norm_inp

        self.input_dim = input_dim

        self.fs = fs

        self.N_cnn_layer = len(num_filters)

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_layer_norm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batch_norm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_layer):

            N_filter = int(self.cnn_N_filter[i])
            len_filter = int(self.cnn_len_filter[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm(
                    [N_filter, int(
                        (current_input - self.cnn_len_filter[i] + 1) /
                        self.cnn_max_pool_len[i]
                    )]
                )
            )

            self.bn.append(
                nn.BatchNorm1d(
                    N_filter, int(
                        (current_input - self.cnn_len_filter[i] + 1) /
                        self.cnn_max_pool_len[i]
                    ), momentum=0.05
                )
            )

            if i == 0:
                self.conv.append(
                    SincConv(
                        self.cnn_N_filter[0], self.cnn_len_filter[0], self.fs
                    )
                )

            else:
                self.conv.append(
                    nn.Conv1d(
                        self.cnn_N_filter[i - 1], self.cnn_N_filter[i],
                        self.cnn_len_filter[i]
                    )
                )

            current_input = int(
                (current_input - self.cnn_len_filter[i] + 1) /
                self.cnn_max_pool_len[i]
            )

        self.out_dim = current_input * N_filter

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_layer_norm_inp):
            x = self.ln0(x)

        if bool(self.cnn_use_batch_norm_inp):
            x = self.bn0(x)

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_layer):

            if self.cnn_use_layer_norm[i]:
                if i == 0:
                    x = self.drop[i](
                        self.act[i](
                            self.ln[i](
                                F.max_pool1d(
                                    torch.abs(self.conv[i](x)),
                                    self.cnn_max_pool_len[i]
                                )
                            )
                        )
                    )
                else:
                    x = self.drop[i](
                        self.act[i](
                            self.ln[i](
                                F.max_pool1d(
                                    self.conv[i](x), self.cnn_max_pool_len[i]
                                )
                            )
                        )
                    )

            if self.cnn_use_batch_norm[i]:
                x = self.drop[i](
                    self.act[i](
                        self.bn[i](
                            F.max_pool1d(
                                self.conv[i](x), self.cnn_max_pool_len[i]
                            )
                        )
                    )
                )

            if (
                self.cnn_use_batch_norm[i] is False
                and self.cnn_use_layer_norm[i] is False
            ):
                x = self.drop[i](
                    self.act[i](
                        F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])
                    )
                )

        x = x.view(batch, -1)

        return x

    @classmethod
    def from_options(cls, options: Dict):
        instance = cls(
            input_dim=options['input_dim'],
            fs=options['fs'],
            num_filters=options['cnn_N_filter'],
            filter_sizes=options['cnn_len_filter'],
            max_pool_sizes=options['cnn_max_pool_len'],
            cnn_use_layer_norm_inp=options['cnn_use_layer_norm_inp'],
            cnn_use_batch_norm_inp=options['cnn_use_batch_norm_inp'],
            cnn_use_layer_norm=options['cnn_use_layer_norm'],
            cnn_use_batch_norm=options['cnn_use_batch_norm'],
            cnn_activations=['cnn_act'],
            cnn_drops=options['cnn_drop']
        )
        return instance