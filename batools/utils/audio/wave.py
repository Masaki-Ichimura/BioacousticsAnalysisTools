import torch
import torchaudio
from torchaudio.sox_effects import apply_effects_tensor
from typing import Union


def load_wave(file_name: str):
    data, fs = torchaudio.load(file_name, normalize=True)
    return data, fs

def metadata_wave(file_name: str):
    metadata = torchaudio.info(file_name)
    params = [
        'num_frames', 'num_channels', 'sample_rate', 'bits_per_sample', 'encoding'
    ]
    return {param: getattr(metadata, param) for param in params}

def save_wave(
    file_name: str, src: torch.Tensor, sample_rate: int, normalization: Union[str, bool]=False
):
    if normalization:
        if normalization == 'ch':
            src = torch.cat(
                [
                    apply_effects_tensor(
                        ch_sig[None], sample_rate, [['gain', '-n']], channels_first=True
                    )[0]
                    for ch_sig in src
                ],
                dim=0
            )
        else:
            src, _ = apply_effects_tensor(
                src, sample_rate, [['gain', '-n']], channels_first=True
            )

    torchaudio.save(file_name, src, sample_rate, channels_first=True)
