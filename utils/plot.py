import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt


def show_spec(waveform, fs, n_fft=2048, **kw):
    waveform = waveform if waveform.ndim == 1 else waveform.mean(0)
    amp_np = torch.stft(
        waveform, n_fft=n_fft, return_complex=True
    ).abs().numpy()

    plt_obj = librosa.display.specshow(
        librosa.amplitude_to_db(amp_np),
        sr=fs,
        y_axis='log', x_axis='time',
        **kw
    )
    return plt_obj

def show_wav(waveform, fs, ax=None):
    waveform = waveform if waveform.ndim == 1 else waveform.mean(0)
    t_len = waveform.shape[0]
    t = torch.linspace(1/fs, t_len/fs, t_len)

    if ax:
        ax.plot(t, waveform)
        return ax
    else:
        plt_obj = plt.plot(t, waveform)
        return plt_obj
