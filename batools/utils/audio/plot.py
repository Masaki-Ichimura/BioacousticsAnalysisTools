import torch

from librosa import amplitude_to_db
from librosa.display import specshow, waveshow


def show_spec(waveform, fs, n_fft, hop_length=None, **kwargs):
    waveform = waveform if waveform.ndim == 1 else waveform.mean(0)
    hop_length = n_fft // 4 if not hop_length else hop_length
    db_np = amplitude_to_db(
        torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True
        ).abs().numpy()
    )
    plt_obj = specshow(
        db_np, sr=fs, y_axis='log', x_axis='time', n_fft=n_fft, hop_length=hop_length, **kwargs
    )
    return plt_obj

def show_wave(waveform, fs, **kwargs):
    waveform = waveform if waveform.ndim == 1 else waveform.mean(0)
    plt_obj = waveshow(waveform.numpy(), sr=fs, **kwargs)
    return plt_obj
