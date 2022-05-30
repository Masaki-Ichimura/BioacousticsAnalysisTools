import torch
import matplotlib.pyplot as plt
from librosa import amplitude_to_db
from librosa.display import specshow, waveshow


def show_spec(waveform, fs, n_fft, **kwargs):
    waveform = waveform if waveform.ndim == 1 else waveform.mean(0)
    db_np = amplitude_to_db(
        torch.stft(
            waveform, n_fft=n_fft, return_complex=True
        ).abs().numpy()
    )
    plt_obj = specshow(
        db_np, sr=fs, y_axis='log', x_axis='time', **kwargs
    )
    return plt_obj

def show_wave(waveform, fs, **kwargs):
    waveform = waveform if waveform.ndim == 1 else waveform.mean(0)
    plt_obj = waveshow(waveform.numpy(), sr=fs, **kwargs)
    return plt_obj
