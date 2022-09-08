import torch
from torchaudio.sox_effects import apply_effects_tensor

from typing import Union, List


def apply_freq_mask(
    signal: torch.Tensor, sample_rate: int,
    freq_low: Union[int, None]=None, freq_high: Union[int, None]=None
):
    freq_min, freq_max = 0, sample_rate//2

    freq_low = max(freq_min, freq_low) if freq_low else freq_min
    if freq_low == freq_min:
        freq_low = ''
    else:
        freq_low = str(int(freq_low))

    freq_high = min(freq_max, freq_high) if freq_high else freq_max
    if freq_high == freq_max:
        freq_high = ''
    else:
        freq_high = str(int(freq_high))

    if freq_high:
        effect = ['sinc', f'{freq_low}-{freq_high}']
    elif freq_low:
        effect = ['sinc', f'{freq_low}']
    else:
        effect = None

    if effect:
        if signal.ndim == 1:
            signal = signal[None]
        signal, _ = apply_effects_tensor(signal, sample_rate, [effect])

    return signal

def extract_from_section(
    signal: torch.Tensor, sample_rate: int, section_ms: List[int]
):
    start_idx, end_idx = [int(t_ms/1000*sample_rate) for t_ms in section_ms]
    extracted = signal[..., start_idx:end_idx]
    return extracted
