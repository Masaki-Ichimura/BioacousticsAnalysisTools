"""
This module is based on pydub (MIT Licence)
    - https://github.com/jiaaro/pydub/blob/master/pydub/silence.py

NOTE:
    - signal xnt is normalized torch.Tensor (channel, time).
        - if you want to use NOT normalized signal,
            1. in code -> max_possible_amplitude value
"""
import torch
import itertools
from math import log
from typing import Union


def db_to_float(db, using_amplitude=True):
    db = float(db)
    if using_amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)

def ratio_to_db(ratio, val2=None, using_amplitude=True):
    ratio = float(ratio)

    if val2 is not None:
        ratio = ratio / val2

    if ratio == 0:
        return -float('inf')

    if using_amplitude:
        return 20 * log(ratio, 10)
    else:  # using power
        return 10 * log(ratio, 10)

def msec_to_index(msec, sample_rate):
    return int(msec/1000*sample_rate)

def rms(xnt):
    return (xnt**2).mean().sqrt()

def dBFS(xnt):
    rms_ = rms(xnt)
    if not rms_:
        return -float('inf')
    return ratio_to_db(rms_ / 1.)


# 仮置き
def silence_removal(
    signal: torch.Tensor, sample_rate: int,
    min_silence_ms: int=1000, seek_ms: int=1, threshold: Union[float, int]=-16, return_prob: bool=False
):
    silent_sections, prob_dict = detect_silence(
        signal, sample_rate, min_silence_ms, threshold, seek_ms, return_prob=True
    )
    sig_len = 1000*round(signal.shape[-1]/sample_rate)

    if not silent_sections:
        return [[0, sig_len]]

    if silent_sections[0][0] == 0 and silent_sections[0][1] == sig_len:
        if return_prob:
            return [], prob_dict
        else:
            return []

    prev_end_i = 0
    nonsilent_sections = []
    for start_i, end_i in silent_sections:
        nonsilent_sections.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != sig_len:
        nonsilent_sections.append([prev_end_i, sig_len])

    if nonsilent_sections[0] == [0, 0]:
        nonsilent_sections.pop(0)

    if return_prob:
        return nonsilent_sections, prob_dict
    else:
        return nonsilent_sections

def detect_silence(
    xnt,
    sample_rate,
    min_silence_len=1000, silence_thresh=-16, seek_step=1, return_prob=False
):

    sig_len = 1000*round(xnt.shape[-1]/sample_rate)

    if sig_len < min_silence_len:
        return []

    silence_thresh = db_to_float(silence_thresh) * 1.

    silence_starts = []

    last_slice_start = sig_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    rms_list = [
        rms(xnt[
            ...,
            msec_to_index(msec, sample_rate): \
            min(
                msec_to_index(msec, sample_rate)+msec_to_index(min_silence_len, sample_rate),
                xnt.shape[-1]
            )
        ])
        for msec in slice_starts
    ]

    probability = torch.stack(rms_list)

    _ = [
        silence_starts.append(msec)
        for msec, rms in zip(slice_starts, rms_list)
        if rms <= silence_thresh
    ]

    if not silence_starts:
        return []

    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([
                current_range_start, prev_i + min_silence_len
            ])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start, prev_i+min_silence_len])

    if return_prob:
        prob_dict = {
            'probability': probability,
            'threshold': silence_thresh
        }
        return silent_ranges, prob_dict
    else:
        return silent_ranges

def detect_nonsilent(
    xnt,
    sample_rate,
    min_silence_len=1000, silence_thresh=-16, seek_step=1
):

    silent_ranges = detect_silence(
        xnt, sample_rate, min_silence_len, silence_thresh, seek_step
    )
    sig_len = 1000*round(xnt.shape[-1]/sample_rate)

    if not silent_ranges:
        return [[0, sig_len]]

    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == sig_len:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != sig_len:
        nonsilent_ranges.append([prev_end_i, sig_len])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges


def split_on_silence(
    xnt,
    sample_rate,
    min_silence_len=1000, silence_thresh=-16, keep_silence=100, seek_step=1
):

    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    sig_len = 1000*round(xnt.shape[-1]/sample_rate)

    if isinstance(keep_silence, bool):
        keep_silence = sig_len if keep_silence else 0

    output_ranges = [
        [ start - keep_silence, end + keep_silence ]
        for (start, end) in detect_nonsilent(
            xnt, sample_rate, min_silence_len, silence_thresh, seek_step
        )
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return [
        xnt[
            :,
            msec_to_index(max(start, 0), sample_rate) : \
            msec_to_index(min(end, sig_len), sample_rate)
        ]
        for start, end in output_ranges
    ]


def detect_leading_silence(
    xnt,
    sample_rate,
    silence_threshold=-50.0, chunk_size=10
):
    trim_ms = 0
    sig_len = 1000*round(xnt.shape[-1]/sample_rate)

    assert chunk_size > 0

    while dBFS(xnt[
        :,
        max(msec_to_index(trim_ms, sample_rate), 0) : \
        min(msec_to_index(trim_ms+chunk_size, sample_rate), sig_len)
    ]) < silence_threshold and trim_ms < sig_len:
        trim_ms += chunk_size

    return min(trim_ms, sig_len)
