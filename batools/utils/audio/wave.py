import torch
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor


def load_wave(file_name: str):
    data, fs = torchaudio.load(file_name, normalize=True)
    return data, fs

def metadata_wave(file_name: str):
    metadata = torchaudio.info(file_name)
    params = [
        'num_frames', 'num_channels', 'sample_rate', 'bits_per_sample', 'encoding'
    ]
    return {param: getattr(metadata, param) for param in params}

def save_wave(file_name: str, src: torch.Tensor, sample_rate: int, normalize: bool=True):
    if normalize:
        src, sample_rate = apply_effects_tensor(
            src, sample_rate, [['gain', '-n']], channels_first=True
        )

    torchaudio.save(file_name, src, sample_rate, channels_first=True)
