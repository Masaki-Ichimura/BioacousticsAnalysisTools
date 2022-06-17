import torch
import torchaudio

from typing import Union


def freq_mask(
    sample_rate: int, n_fft: int,
    freq_low: Union[float, int]=0, freq_high: Union[float, int]=float('inf')
):
    k = torch.fft.rfftfreq(n_fft) * sample_rate
    mask = torch.logical_or(k < freq_low, k > freq_high)
    return mask
