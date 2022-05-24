import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt


def show_spec(waveform, fs, n_fft=2048,**kw):
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

def show_wav(waveform):
    waveform = waveform if waveform.ndim == 1 else waveform.mean(0)
    plt_obj = plt.plot(waveform)
    return plt_obj
