import torch as th
import torch.nn as nn
import torch.nn.functional as tf
from typing import Optional
import math
import numpy as np
import pdb


def perturb_speed(wav: th.Tensor, weight: th.Tensor):
    """
    Do speed perturb
    Args:
        wav (Tensor): N x S
        weight (Tensor): dst_sr x src_sr x K
    Return
        wav (Tensor): N x (N/src_sr)*dst_sr
    """
    #pdb.set_trace()
    _, src_sr, K = weight.shape
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    N, S = wav.shape
    num_blocks = S // src_sr
    if num_blocks == 0:
        raise RuntimeError(
            f"Input wav is too short to be perturbed, length = {S}")
    # N x B x sr
    wav = wav[:, :num_blocks * src_sr].view(N, num_blocks, -1)
    # N x src_sr x B
    wav = wav.transpose(1, 2)
    # N x dst_sr x B
    wav = tf.conv1d(wav.float(), weight.float(), padding=(K - 1) // 2)
    # N x B x dst_sr
    wav = wav.transpose(1, 2).contiguous()
    # N x B*dst_sr
    return wav.view(N, -1)


def speed_perturb_filter(src_sr: int,
                         dst_sr: int,
                         cutoff_ratio: float = 0.95,
                         num_zeros: int = 64) -> th.Tensor:
    """
    Return speed perturb filters, reference:
        https://github.com/danpovey/filtering/blob/master/lilfilter/resampler.py
    Args:
        src_sr: sample rate of the source signal
        dst_sr: sample rate of the target signal
    Return:
        weight (Tensor): coefficients of the filter
    """
    if src_sr == dst_sr:
        raise ValueError(
            f"src_sr should not be equal to dst_sr: {src_sr}/{dst_sr}")
    gcd = math.gcd(src_sr, dst_sr)
    src_sr = src_sr // gcd
    dst_sr = dst_sr // gcd
    if src_sr == 1 or dst_sr == 1:
        raise ValueError("do not support integer downsample/upsample")
    zeros_per_block = min(src_sr, dst_sr) * cutoff_ratio
    padding = 1 + int(num_zeros / zeros_per_block)
    # dst_sr x src_sr x K
    times = (np.arange(dst_sr)[:, None, None] / float(dst_sr) -
             np.arange(src_sr)[None, :, None] / float(src_sr) -
             np.arange(2 * padding + 1)[None, None, :] + padding)
    window = np.heaviside(1 - np.abs(times / padding),
                          0.0) * (0.5 + 0.5 * np.cos(times / padding * math.pi))
    weight = np.sinc(
        times * zeros_per_block) * window * zeros_per_block / float(src_sr)
    return th.tensor(weight, dtype=th.float32)



class SpeedPerturb(nn.Module):
    """
    Transform layer for performing speed perturb
    Args:
        sr: sample rate of source signal
        perturb: speed perturb factors
    """

    def __init__(self, sr: int = 16000, perturb: str = "0.9,1.0,1.1") -> None:
        super(SpeedPerturb, self).__init__()
        self.sr = sr
        self.factor_str = perturb
        dst_sr = [int(factor * sr) for factor in map(float, perturb.split(","))]
        if not len(dst_sr):
            raise ValueError("No perturb options for doing speed perturb")
        # N x dst_sr x src_sr x K
        self.weights = nn.ParameterList([
            nn.Parameter(speed_perturb_filter(sr, fs), requires_grad=False)
            for fs in dst_sr
            if fs != sr
        ])
        self.last_weight = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sr={self.sr}, factor={self.factor_str})"

    def output_length(self,
                      inp_len: Optional[th.Tensor]) -> Optional[th.Tensor]:
        """
        Compute output length after speed perturb
        """
        if self.last_weight is None:
            return inp_len
        if inp_len is None:
            return None
        dst_sr, src_sr, _ = self.last_weight.shape
        return (inp_len // src_sr) * dst_sr

    def forward(self, wav: th.Tensor) -> th.Tensor:
        """
        Args:
            wav (Tensor): input signal, N x ... x S
        Return:
            wav (Tensor): output signal, N x ... x S
        """
        self.last_weight = None
        if not self.training:
            return wav
#         if wav.dim() != 2:
#             raise RuntimeError(f"Now only supports 2D tensor, got {wav.dim()}")
        # 1.0, do not apply speed perturb
        choice = th.randint(0, len(self.weights) + 1, (1,)).item()
        if choice == len(self.weights):
            return wav
        else:
            self.last_weight = self.weights[choice]
            return perturb_speed(wav, self.last_weight)


    