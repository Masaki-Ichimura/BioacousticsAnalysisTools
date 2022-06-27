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

def extract_from_sections(
    signal: torch.Tensor, sample_rate: int, extract_ms_sections: list
):
    extracted = [
        signal[..., int(sec[0]/1000*sample_rate):int(sec[1]/1000*sample_rate)]
        for sec in extract_ms_sections
    ]
    return extracted
