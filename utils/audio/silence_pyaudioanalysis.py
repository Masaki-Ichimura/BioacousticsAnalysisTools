"""
    This module is based on PyAudioAnalysis (Apache Licence)
        - https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioSegmentation.py
"""
import torch
from torchaudio.functional import create_dct

from typing import Union

from sklearn.svm import SVC

EPS = 1e-15


def silence_removal(
    signal,
    sample_rate: int, win_msec: int, seek_msec: int,
    freq_low: Union[float, int]=0, freq_high: Union[float, int]=float('inf'),
    smooth_window_msec: int=500, min_duration_msec: Union[None, int]=200,
    weight: float=.5
):
    weight = max(0, min(1, weight))

    # signal: torch.tensor (channel, t) -> (t,)
    signal = signal.mean(0)

    window = win_msec*sample_rate//1000
    seek = seek_msec*sample_rate//1000
    n_fft = window//2

    fbank, freq = mfcc_filter_banks(sample_rate, n_fft)
    freq = freq[1:-1]
    fbank = fbank[torch.logical_and(freq>=freq_low, freq<=freq_high)]
    def feature_extraction(frame, fft_magnitude, fft_magnitude_previous):
        feature = [
            zero_crossing_rate(frame),
            energy(frame),
            energy_entropy(frame),
            *spectral_centroid_spread(fft_magnitude, sample_rate),
            spectral_entropy(fft_magnitude),
            spectral_flux(fft_magnitude, fft_magnitude_previous),
            spectral_rolloff(fft_magnitude, .9),
            *mfcc(fft_magnitude, fbank, -1),
            *chroma_features(fft_magnitude, sample_rate, n_fft)
        ]
        return torch.stack(feature)

    features = []
    current_pos = 0
    while 1:
        x = signal[current_pos:current_pos+window]

        if x.numel() == 0:
            break
        else:
            if len(x) != window:
                x = torch.nn.functional.pad(x, [0, window-len(x)])

            prev_pos = current_pos-seek
            if prev_pos < 0:
                x_prev = x
            else:
                x_prev = signal[prev_pos:prev_pos+window]

            fft_mag = torch.fft.fft(x)[:n_fft].abs()/n_fft

            if current_pos == 0:
                # not required to copy tensor
                fft_mag_prev = fft_mag
                fft_mag_pprev = fft_mag

            feature = feature_extraction(x, fft_mag, fft_mag_prev)

            # delta features
            feature_prev = feature_extraction(x_prev, fft_mag_prev, fft_mag_pprev)
            delta = feature-feature_prev

            features.append(torch.cat((feature, delta)))

            # not required to copy tensor
            fft_mag_prev = fft_mag
            fft_mag_pprev = fft_mag_prev

            current_pos += seek

    features = torch.stack(features)
    features_norm = (features-features.mean(0))/features.std(0)
    energies = features[:, 1]
    en_val, en_indices = energies.sort()

    n_split = len(features)
    st_windows_fraction = n_split//10

    split_indices = torch.arange(n_split)
    low_indices = split_indices[
        energies<=(en_val[:st_windows_fraction].mean() + 1e-15)
    ]
    high_indices = split_indices[
        energies>=(en_val[-st_windows_fraction:].mean() + 1e-15)
    ]

    low_features = features[low_indices]
    high_features = features[high_indices]

    datas = torch.cat([low_features, high_features])
    labels = torch.cat([
        torch.zeros(low_features.shape[0]), torch.ones(high_features.shape[0])
    ])

    # TODO: use SVM (or other classifier model) for pytorch
    model = SVC(C=1., kernel='linear', probability=True)
    model.fit(datas, labels)

    # method: 1
    # prob = smooth_moving_avg(
    #     torch.from_numpy(model.predict_proba(features)).T,
    #     smooth_window_msec//seek_msec
    # ).T
    # prob = torch.where(prob.softmax(-1)[:, 0]>.5, 0, 1)

    # method: 2 (based on pyAudioAnalysis)
    prob = smooth_moving_avg(
        torch.from_numpy(model.predict_proba(features))[:, 0],
        smooth_window_msec//seek_msec
    )
    prob_sort = prob.sort().values
    nt = len(prob)//10
    prob_thr = (1-weight)*prob_sort[:nt].mean()+weight*prob_sort[-nt:].mean()
    prob = torch.where(prob>prob_thr, 0, 1)

    # clustering
    start_t = prob == 0
    if start_t.any():
        idx = torch.where(start_t)[0][0]
        while 1:
            high_t = prob[idx:] == 1
            if not high_t.any():
                break

            next_idx = idx + torch.where(high_t)[0][0].item() - 1
            if next_idx - idx <= 3:
                prob[list(range(idx, next_idx+1))] = 1

            low_t = prob[next_idx:] == 0
            if not low_t.any():
                break

            idx = next_idx + torch.where(low_t)[0][0].item()

    idx = 0
    nonsilent_sections = []
    while 1:
        start_t = prob[idx:] == 1
        if not start_t.any():
            break
        else:
            start_idx = idx + torch.where(start_t)[0][0].item()

            end_t = prob[start_idx:] == 0
            if not end_t.any():
                end_idx = len(prob)
            else:
                end_idx = start_idx + torch.where(end_t)[0][0].item()-1

            nonsilent_sections.append([start_idx*seek_msec, end_idx*seek_msec])

        idx = end_idx+1

    if min_duration_msec:
        nonsilent_sections = [
            sec for sec in nonsilent_sections if sec[1]-sec[0] > min_duration_msec
        ]

    return nonsilent_sections


"""
    General utility functions
"""
def smooth_moving_avg(x, n_win=11):
    if x.ndim == 1:
        x = x[None, None]
    elif x.ndim == 2:
        x = x[:, None]
    w = torch.ones(n_win, dtype=x.dtype)[None, None]

    return torch.conv1d(x, w, bias=None, padding='valid').squeeze() / n_win


"""
    Feature extraction
        - https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/ShortTermFeatures.py
"""
def zero_crossing_rate(frame):
    return frame.sign().diff().abs().sum() / ((len(frame)-1.) * 2)

def energy(frame):
    return (frame**2).mean()

def energy_entropy(frame, n_short_blocks=10):
    frame_energy = (frame**2).sum()
    frame_length = len(frame)

    sub_win_len = int(frame_length / n_short_blocks)

    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[:sub_win_len * n_short_blocks]

    sub_wins = frame.reshape(n_short_blocks, sub_win_len).T
    s = (sub_wins ** 2).sum(0) / (frame_energy + EPS)
    return -(s * (s + EPS).log2()).sum()


def spectral_centroid_spread(fft_magnitude, sample_rate):
    n_fft = len(fft_magnitude)
    ind = (torch.arange(1, n_fft+1)) * (sample_rate / (2.*n_fft))

    Xt = fft_magnitude[:]
    Xt_max = Xt.max()
    Xt = Xt / EPS if Xt_max == 0 else Xt / Xt_max

    NUM = (ind * Xt).sum()
    DEN = Xt.sum() + EPS

    centroid = (NUM / DEN)

    spread = ((((ind - centroid) ** 2) * Xt).sum() / DEN).sqrt()

    centroid = 2. * centroid / sample_rate
    spread = 2. * spread / sample_rate

    return centroid, spread

def spectral_entropy(fft_magnitude, n_short_blocks=10):
    n_fft = len(fft_magnitude)
    total_energy = (fft_magnitude ** 2).sum()

    sub_win_len = n_fft//n_short_blocks

    if n_fft != sub_win_len * n_short_blocks:
        fft_magnitude = fft_magnitude[:sub_win_len * n_short_blocks]

    sub_wins = fft_magnitude.reshape(n_short_blocks, sub_win_len).T
    s = (sub_wins ** 2).sum(0) / (total_energy + EPS)
    return -(s * (s + EPS).log2()).sum()

def spectral_flux(fft_magnitude, fft_magnitude_previous):
    fft_std = fft_magnitude/(fft_magnitude + EPS).sum()
    fft_std_prev = fft_magnitude_previous/(fft_magnitude_previous + EPS).sum()
    return ((fft_std - fft_std_prev) ** 2).sum()

def spectral_rolloff(fft_magnitude, c):
    energy = (fft_magnitude ** 2).sum()
    n_fft = len(fft_magnitude)
    threshold = c * energy
    cumulative_sum = (fft_magnitude ** 2).cumsum(0) + EPS
    a = torch.nonzero(cumulative_sum > threshold).squeeze()
    sp_rolloff = a[0]/n_fft if len(a) > 0 else 0.
    return sp_rolloff

def mfcc_filter_banks(
    sample_rate, n_fft,
    freq_low=133.33, linc=200 / 3, logsc=1.0711703,
    num_lin_filt=13, num_log_filt=27
):

    if sample_rate < 8000:
        nlogfil = 5

    num_filt_total = num_lin_filt + num_log_filt

    freq = torch.zeros(num_filt_total + 2)
    freq[:num_lin_filt] = freq_low + torch.arange(num_lin_filt) * linc
    freq[num_lin_filt:] = freq[num_lin_filt - 1] * logsc ** \
                                 torch.arange(1, num_log_filt + 3)
    heights = 2. / (freq[2:] - freq[0:-2])

    fbank = torch.zeros((num_filt_total, n_fft))
    nfreqs = torch.arange(n_fft) / (1. * n_fft) * sample_rate

    for i in range(num_filt_total):
        low_freq = freq[i]
        cent_freq = freq[i + 1]
        high_freq = freq[i + 2]

        lid = torch.arange(
            (low_freq * n_fft / sample_rate).floor() + 1,
            (cent_freq * n_fft / sample_rate).floor() + 1,
            dtype=torch.long
        )
        lslope = heights[i] / (cent_freq - low_freq)
        rid = torch.arange(
            (cent_freq * n_fft / sample_rate).floor() + 1,
            (high_freq * n_fft / sample_rate).floor() + 1,
            dtype=torch.long
        )
        rslope = heights[i] / (high_freq - cent_freq)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_freq)
        fbank[i][rid] = rslope * (high_freq - nfreqs[rid])

    return fbank, freq

def mfcc(fft_magnitude, fbank, num_mfcc_feats):
    if num_mfcc_feats == -1:
        num_mfcc_feats = None
    mspec = (fbank@fft_magnitude + EPS).log10()
    dct_ceps = create_dct(n_mfcc=len(mspec), n_mels=len(mspec), norm='ortho')
    return (dct_ceps.T@mspec)[:num_mfcc_feats]

def chroma_features_init(n_fft, sample_rate):
    freq = torch.arange(1, n_fft+1)*(sample_rate/(2 * n_fft))
    cp = 27.50
    num_chroma = (12*(freq / cp).log2()).round().type(torch.long)

    num_freq_per_chroma = torch.zeros(n_fft)

    unique_chroma, count_chroma = torch.unique(num_chroma, return_counts=True)

    for u, c in zip(unique_chroma, count_chroma):
        num_freq_per_chroma[num_chroma==u] = c

    return num_chroma, num_freq_per_chroma

def chroma_features(fft_magnitude, sample_rate, n_fft):
    num_chroma, num_freq_per_chroma = chroma_features_init(n_fft, sample_rate)
    # chroma_names = [
    #     'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'
    # ]
    spec = fft_magnitude ** 2
    C = torch.zeros(n_fft)
    if num_chroma.max() < n_fft:
        C[num_chroma] = spec
        C /= num_freq_per_chroma[num_chroma]
    else:
        I = int(torch.nonzero(num_chroma > n_fft)[0].item())
        C[num_chroma[:I - 1]] = spec
        C /= num_freq_per_chroma
    final_matrix = torch.zeros((12, 1))
    # ceil(x/y) -> -(-x//y)
    newD = -(-n_fft//12) * 12
    C2 = torch.zeros(newD)
    C2[:n_fft] = C
    C2 = C2.reshape(-1, 12)

    final_matrix = C2.sum(0)

    spec_sum = spec.sum()
    if spec_sum == 0:
        final_matrix /= EPS
    else:
        final_matrix /= spec_sum

    return final_matrix
