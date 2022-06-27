"""
    This module is based on PyAudioAnalysis (Apache Licence)
        - https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioSegmentation.py
"""
import torch
from torchaudio.functional import create_dct
from sklearn.svm import SVC

from typing import Union

EPS = 1e-15


def silence_removal(
    signal: torch.Tensor, sample_rate: int, win_ms: int, seek_ms: int,
    freq_low: Union[float, int]=0., freq_high: Union[float, int]=float('inf'),
    mask_ms_sections: list=None,
    smooth_window_ms: int=500, broaden_section_ms: int=0,
    min_nonsilence_ms: Union[int, None]=200,
    weight: float=.5, return_prob: bool=False
):
    """
    Getting non-silent sections for audio signal.

    Parameters
    ----------
    signal : torch.Tensor
        Audio signal (ch, t) or (t,)
    win_ms : int
        Window length in ms.
    seek_ms : int
        Seek length for window in ms.
    freq_low : float or int, default 0.
        Lowest frequency of MFCC feature.
    freq_high : float or int, default float('inf')
        Highest frequency of MFCC feature.
    mask_ms_sections : list or None, default None
        Mask sections for classification of silent or nonsilent section of signal.
    smooth_window_ms : int, default 500
        Using to smooth trained classifier probability.
    broaden_section_ms : int, default 0
        Broaden range of non-silent sections in ms.
    min_nonsilence_ms : int or None, default 200
        Minimum duration of non-silent sections.
    weight : float, default .5
        The higher, the more strict to classify silence.
    return_prob : bool, default False
        Return trained classifier probabilities and threshold.

    Returns
    -------
    nonsilent_sections : list
        List of Non-silent sections.
    prob_dict : dict
        Dictionary of trained classifier Probabilities and threshold
    """
    weight = max(0, min(1, weight))

    if signal.ndim == 2:
        signal = signal.mean(0)

    window = win_ms * sample_rate // 1000
    seek = seek_ms * sample_rate // 1000
    n_fft = window // 2

    fbank, freq = mfcc_filter_banks(sample_rate, n_fft)
    freq = freq[1:-1]
    fbank = fbank[torch.logical_and(freq>=freq_low, freq<=freq_high)]

    num_chroma, num_freq_per_chroma = chroma_features_init(sample_rate, n_fft)

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
            *chroma_features(fft_magnitude, sample_rate, num_chroma, num_freq_per_chroma)
        ]
        return torch.stack(feature)

    features = []
    current_pos = 0
    while 1:
        x = signal[current_pos:current_pos+window]

        if x.numel() == 0:
            break
        else:
            if x.size(0) != window:
                x = torch.nn.functional.pad(x, [0, window-x.size(0)])

            fft_mag = torch.fft.rfft(x)[:n_fft].abs() / n_fft

            if current_pos == 0:
                feature = feature_extraction(x, fft_mag, fft_mag)
                feature_prev = feature
            else:
                feature = feature_extraction(x, fft_mag, fft_mag_prev)

            # current feature + delta feature
            features.append(torch.cat([feature, feature-feature_prev]))

            # NOT required to copy tensor
            fft_mag_prev = fft_mag
            feature_prev = feature

            current_pos += seek

    features = torch.stack(features)
    features_norm = (features - features.mean(0)) / features.std(0)
    energies = features[:, 1]
    if mask_ms_sections:
        mask_indices = [
            set(range(sec[0]//seek_ms, sec[1]//seek_ms+1))
            for sec in mask_ms_sections
        ]
        tmp = mask_indices[0]
        _ = [[tmp.add(idx) for idx in indices] for indices in mask_indices[1:]]
        mask_indices = tmp
        energies = energies[
            [idx for idx in range(energies.size(0)) if idx not in mask_indices]
        ]
    energies_sort = energies.sort().values

    n_split = features.size(0)
    st_windows_fraction = n_split // 10

    split_indices = torch.arange(n_split)
    low_indices = split_indices[
        energies <= (energies_sort[:st_windows_fraction].mean() + 1e-15)
    ]
    high_indices = split_indices[
        energies >= (energies_sort[-st_windows_fraction:].mean() + 1e-15)
    ]

    datas = torch.cat([features[low_indices], features[high_indices]])
    labels = torch.cat([
        torch.zeros(low_indices.size(0)), torch.ones(high_indices.size(0))
    ])

    model = SVC(C=1., kernel='linear', probability=True)
    model.fit(datas.numpy(), labels.numpy())

    # method: 1
    # pr_org = smooth_moving_avg(
    #     torch.from_numpy(model.predict_proba(features)).T,
    #     smooth_window_ms // seek_ms
    # ).T
    # pr = torch.where(pr_org.softmax(-1)[:, 0]>.5, 0, 1)

    # method: 2 (based on pyAudioAnalysis)
    pr_org = smooth_moving_avg(
        torch.from_numpy(model.predict_proba(features))[:, 0],
        smooth_window_ms // seek_ms
    )
    pr_sort = pr_org.sort().values
    nt = pr_org.size(0) // 10
    pr_thr = (1-weight)*pr_sort[:nt].mean() + weight*pr_sort[-nt:].mean()

    nonsilent_sections = segmentation(
        pr_org, pr_thr, seek_ms,
        clustering=True,
        broaden_section_num=broaden_section_ms, enable_merge=True,
        min_duration_num=min_nonsilence_ms
    )

    if return_prob:
        prob_dict = {
            'probability': pr_org,
            'threshold': pr_thr.item()
        }
        return nonsilent_sections, prob_dict
    else:
        return nonsilent_sections


"""
    Segmentation funcrion
        1. original probability to binary using threshold
        2. clustering
        3. make non-silent sections from binary probability
        4. broaden range of sections
        5. remove section less than minimum duration
"""
def segmentation(
    pr, threshold, seek_num,
    clustering=False, broaden_section_num=0, enable_merge=True,
    min_duration_num=None,
):
    pr = torch.where(pr > threshold, 0, 1)

    # clustering
    if clustering:
        start_t = (pr == 0)
        if start_t.any():
            idx = torch.where(start_t)[0][0]
            while 1:
                high_t = (pr[idx:] == 1)
                if not high_t.any():
                    break

                next_idx = idx + torch.where(high_t)[0][0].item()
                if next_idx - idx <= 3:
                    pr[list(range(idx, next_idx))] = 1

                low_t = (pr[next_idx:] == 0)
                if not low_t.any():
                    break

                idx = next_idx + torch.where(low_t)[0][0].item()

    idx = 0
    nonsilent_sections = []
    while 1:
        start_t = (pr[idx:] == 1)
        if not start_t.any():
            break
        else:
            start_idx = idx + torch.where(start_t)[0][0].item()

            end_t = (pr[start_idx:] == 0)
            if not end_t.any():
                end_idx = pr.size(0)
            else:
                end_idx = start_idx + torch.where(end_t)[0][0].item() - 1

            nonsilent_sections.append([start_idx*seek_num, end_idx*seek_num])

        idx = end_idx + 1

    if broaden_section_num:
        broaden_sections = [
            [max(0, sec[0]-broaden_section_num), sec[1]+broaden_section_num]
            for sec in nonsilent_sections
        ]
        tmp = []
        if enable_merge:
            p = broaden_sections.pop(0)
            while 1:
                if broaden_sections:
                    q = broaden_sections.pop(0)
                    if p[1]>=q[0]:
                        p = [p[0], q[1]]
                    else:
                        tmp.append(p)
                        p = q
                else:
                    if tmp[-1][1]>=q[0]:
                        tmp[-1] = [tmp[-1][0], q[1]]
                    else:
                        tmp.append(q)
                    break

        nonsilent_sections = tmp

    if min_duration_num:
        nonsilent_sections = [
            sec for sec in nonsilent_sections if sec[1]-sec[0] > min_duration_num
        ]

    return nonsilent_sections


"""
    General utility function
        - https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioSegmentation.py
"""
def smooth_moving_avg(x, n_win=11):
    if n_win < 3:
        return x

    if x.ndim == 1:
        x = x[None]
    elif x.ndim > 2:
        raise ValueError('')

    s = torch.hstack([
        2*x[:, 0, None] - x.flip(-1)[:, -n_win:],
        x,
        2*x[:, -1, None] - x.flip(-1)[:, :n_win-1]
    ])[:, None]
    w = torch.ones((1, 1, n_win), dtype=s.dtype) / n_win

    y = torch.conv1d(s, w, bias=None, padding='same').squeeze()
    return y[..., n_win-1:-n_win]


"""
    Feature extraction
        - https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/ShortTermFeatures.py
"""
def zero_crossing_rate(frame):
    return frame.sign().diff().abs().sum() / ((len(frame) - 1) * 2)

def energy(frame):
    return frame.square().mean()

def energy_entropy(frame, n_short_blocks=10):
    frame_length = frame.size(0)
    frame_energy = frame.square().sum()

    sub_win_len = frame_length // n_short_blocks

    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[:sub_win_len*n_short_blocks]

    sub_wins = frame.reshape(n_short_blocks, sub_win_len).T
    s = sub_wins.square().sum(0) / (frame_energy + EPS)
    return -(s * (s + EPS).log2()).sum()

def spectral_centroid_spread(fft_magnitude, sample_rate):
    n_fft = fft_magnitude.size(0)
    ind = torch.arange(1, n_fft+1) * sample_rate / (2 * n_fft)

    Xt = fft_magnitude / max(EPS, fft_magnitude.max())

    NUM, DEN = (ind * Xt).sum(), Xt.sum() + EPS

    centroid = NUM / DEN
    spread = (((ind - centroid).square() * Xt).sum() / DEN).sqrt()

    centroid = 2 * centroid / sample_rate
    spread = 2 * spread / sample_rate

    return centroid, spread

def spectral_entropy(fft_magnitude, n_short_blocks=10):
    n_fft = fft_magnitude.size(0)
    total_energy = fft_magnitude.square().sum()

    sub_win_len = n_fft // n_short_blocks

    if n_fft != sub_win_len * n_short_blocks:
        fft_magnitude = fft_magnitude[:sub_win_len*n_short_blocks]

    sub_wins = fft_magnitude.reshape(n_short_blocks, sub_win_len).T
    s = sub_wins.square().sum(0) / (total_energy + EPS)
    return -(s * (s + EPS).log2()).sum()

def spectral_flux(fft_magnitude, fft_magnitude_previous):
    fft_std = fft_magnitude / (fft_magnitude + EPS).sum()
    fft_std_prev = fft_magnitude_previous / (fft_magnitude_previous + EPS).sum()
    return (fft_std - fft_std_prev).square().sum()

def spectral_rolloff(fft_magnitude, c):
    n_fft = fft_magnitude.size(0)

    energy = fft_magnitude.square()
    total_energy = energy.sum()

    threshold = c * total_energy
    cumulative_sum = energy.cumsum(0) + EPS
    a = torch.nonzero(cumulative_sum > threshold).squeeze()
    sp_rolloff = a[0] / n_fft if a.size(0) > 0 else 0.
    return sp_rolloff

def mfcc_filter_banks(
    sample_rate, n_fft,
    freq_low=133.33, linc=200 / 3, logsc=1.0711703,
    num_lin_filt=13, num_log_filt=27
):
    num_filt_total = num_lin_filt + num_log_filt

    freq = torch.zeros(num_filt_total+2)
    freq[:num_lin_filt] = freq_low + torch.arange(num_lin_filt) * linc
    freq[num_lin_filt:] = \
        freq[num_lin_filt-1] * logsc ** torch.arange(1, num_log_filt+3)
    heights = 2. / (freq[2:] - freq[:-2])

    fbank = torch.zeros((num_filt_total, n_fft))
    nfreqs = torch.arange(n_fft) * sample_rate / (1 * n_fft)

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
    n = mspec.size(0)
    dct_ceps = create_dct(n_mfcc=n, n_mels=n, norm='ortho')
    return (dct_ceps.T@mspec)[:num_mfcc_feats]

def chroma_features_init(sample_rate, n_fft):
    freq = torch.arange(1, n_fft+1) * sample_rate / (2 * n_fft)
    cp = 27.50
    num_chroma = (12 * (freq / cp).log2()).round().type(torch.long)

    num_freq_per_chroma = torch.zeros(n_fft)

    unique_chroma, count_chroma = torch.unique(num_chroma, return_counts=True)

    for u, c in zip(unique_chroma, count_chroma):
        num_freq_per_chroma[num_chroma == u] = c

    return num_chroma, num_freq_per_chroma

def chroma_features(
    fft_magnitude, sample_rate,
    num_chroma=None, num_freq_per_chroma=None
):
    n_fft = fft_magnitude.size(0)

    if num_chroma is None or num_freq_per_chroma is None:
        num_chroma, num_freq_per_chroma = chroma_features_init(sample_rate, n_fft)
    # chroma_names = [
    #     'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'
    # ]

    spec = fft_magnitude.square()

    C = torch.zeros(n_fft)
    if num_chroma.max() < n_fft:
        C[num_chroma] = spec
        C /= num_freq_per_chroma[num_chroma]
    else:
        I = int(torch.nonzero(num_chroma > n_fft)[0].item())
        C[num_chroma[:I-1]] = spec
        C /= num_freq_per_chroma

    # ceil(x/y) -> -(-x//y)
    C2 = torch.zeros(-(-n_fft//12) * 12)
    C2[:n_fft] = C

    final_matrix = C2.reshape(-1, 12).sum(0) / max(EPS, spec.sum())

    return final_matrix
