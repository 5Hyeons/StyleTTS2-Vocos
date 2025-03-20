from typing import Optional

import math
import random
import numpy as np
import time

import torch
from torch import nn
import torch.nn.functional as F
# from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn import Conv1d
from .utils import init_weights, get_padding

from typing import Optional, Tuple
from scipy.signal import get_window

class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])


    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
            
class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        style_dim: int,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.norm = AdaIN1d(style_dim, dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x, s)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
#             # for normal case

#             # To prevent torch.cumsum numerical overflow,
#             # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
#             # Buffer tmp_over_one_idx indicates the time step to add -1.
#             # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
#             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             cumsum_shift = torch.zeros_like(rad_values)
#             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

#             phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2), 
                                                         scale_factor=1/self.upsample_scale, 
                                                         mode="linear").transpose(1, 2)
    
#             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             cumsum_shift = torch.zeros_like(rad_values)
#             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
    
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale, 
                                                    scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
            
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                             device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
    
def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)

class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class Generator(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        style_dim: int,
        intermediate_dim: int,
        num_layers: int,
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers

        self.m_source = SourceModuleHnNSF(
                    # sampling_rate=24000//gen_istft_hop_size,
                    sampling_rate=24000,
                    upsample_scale=1,
                    harmonic_num=8, voiced_threshod=10)
        # self.f0_upsamp = torch.nn.Upsample(scale_factor=gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.convnext = nn.ModuleList()
                
        for i in range(num_layers):
            # Add Conv1d for noise injection
            self.noise_convs.append(
                nn.Conv1d(1, dim, kernel_size=3, padding=1)
                # nn.Conv1d(gen_istft_n_fft + 2, dim, kernel_size=3, padding=1)
            )
            # Add residual blocks for conditioning noise with style
            self.noise_res.append(
                AdaINResBlock1(dim, 3, [1, 3, 5], style_dim)
                # AdaINResBlock1(dim, 7, [1, 3, 5], style_dim)
            )
            # Add ConvNeXt block
            self.convnext.append(
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    style_dim=style_dim,
                )
            )

        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = ISTFTHead(dim=dim, n_fft=gen_istft_n_fft, hop_length=gen_istft_hop_size, padding="same")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, s, f0) -> torch.Tensor:
        har_source, noi_source, uv = self.m_source(f0.unsqueeze(-1))
        har_source = har_source.transpose(1, 2)
        for i, conv_block in enumerate(self.convnext):
            x_source = self.noise_convs[i](har_source)
            x_source = self.noise_res[i](x_source, s)

            # x = self.reflection_pad(x)
            x = x + x_source
            x = conv_block(x, s)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.stft(x)
        return x

class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y
    
class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        self.filter_length = n_fft
        self.win_length = n_fft
        self.hop_length = hop_length
        self.window = torch.from_numpy(get_window("hann", self.win_length, fftbins=True).astype(np.float32))

        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value 
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    
class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

class Decoder(nn.Module):
    def __init__(self, dim_in=512, style_dim=64, dim_out=80, 
                intermediate_dim=1536,
                num_layers=8,
                gen_istft_n_fft=1024, gen_istft_hop_size=256):
        super().__init__()
        
        self.decode = nn.ModuleList()
        
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))

        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
        )
        
        self.generator = Generator(input_channels=dim_out, dim=dim_in, style_dim=style_dim, 
                                   intermediate_dim=intermediate_dim, num_layers=num_layers, 
                                   gen_istft_n_fft=gen_istft_n_fft, gen_istft_hop_size=gen_istft_hop_size)
        
    def forward(self, asr, F0_curve, N, s):
        if self.training:
            downlist = [0, 3, 7]
            F0_down = downlist[random.randint(0, 2)]
            downlist = [0, 3, 7, 15]
            N_down = downlist[random.randint(0, 3)]
            if F0_down:
                F0_curve = nn.functional.conv1d(F0_curve.unsqueeze(1), torch.ones(1, 1, F0_down).to('cuda'), padding=F0_down//2).squeeze(1) / F0_down
            if N_down:
                N = nn.functional.conv1d(N.unsqueeze(1), torch.ones(1, 1, N_down).to('cuda'), padding=N_down//2).squeeze(1)  / N_down

        
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        
        asr_res = self.asr_res(asr)
        
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        
        x = self.generator(x, s, F0_curve)
        x = x.unsqueeze(1)
        return x
    