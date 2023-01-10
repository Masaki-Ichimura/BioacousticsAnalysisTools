"""
    This module is based on Pyroomacoustics
        - License :
            - MIT License
            - https://github.com/LCAV/pyroomacoustics/blob/pypi-release/LICENSE
        - Original @ fakufaku :
            - https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/doa/doa.py
            - https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/doa/grid.py
"""

import torch
import numpy as np


class DOA(torch.nn.Module):
    def __init__(
        self,
        mic_locs,
        sample_rate,
        n_fft,
        n_src=1,
        r=1.,
        n_grid=360,
        mode='far',
        freq_range=[500.0, 4000.0],
        freq_bins=None,
        freq_hz=None,
        c=343.0,
    ):
        super(DOA, self).__init__()

        self.mic_locs = mic_locs # (xyz, ch)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_grid = n_grid
        self.n_src = n_src
        self.c = c
        self.r = r if mode != 'far' else 1.
        self.mode = mode

        self.max_bin = n_fft//2 + 1

        if freq_bins is not None:
            freq_bins = np.array(freq_bins).astype(int)
        elif freq_hz is not None:
            freq_bins = np.round([f / sample_rate * n_fft for f in freq_hz]).astype(int)
        else:
            freq_range = np.round([f / sample_rate * n_fft for f in freq_range]).astype(int)
            freq_bins = np.arange(*freq_range)

        freq_bins = freq_bins[(0<=freq_bins)&(freq_bins<self.max_bin)]

        self.freq_bins = freq_bins
        self.freq_hz = freq_bins * (sample_rate/n_fft)
        self.num_freq = len(self.freq_bins)

        self.grid = np.array([
            (r*np.cos(rad), r*np.sin(rad), 0)
            for rad in np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        ])

        # k, ch, n_grid
        """
        NOTE:
            nn.Module の内部に格納するパラメータは実数にするのが無難
            .float() や .double() は現時点で complex パラメータをキャストしてくれない
            (.cuda() で GPU に転送してくれるかどうかは試してないのでわからん)
            格納時は実数，実際の計算を行う時に複素に戻すようにする
        """
        self.mode_vec = torch.nn.parameter.Parameter(
            data=torch.view_as_real(torch.from_numpy(self.precompute_mode_vec())),
            requires_grad=False
        )

    def forward(self):
        raise NotImplementedError

    def precompute_mode_vec(self):
        p_x = self.grid[None, None, ..., 0]
        p_y = self.grid[None, None, ..., 1]
        p_z = self.grid[None, None, ..., 2]

        r_x = self.mic_locs[[0], :, None]
        r_y = self.mic_locs[[1], :, None]

        if self.mic_locs.shape[0] == 3:
            r_z = self.mic_locs[[2], :, None]
        else:
            r_z = np.zeros((1, self.mic_locs.shape[1], 1))

        # if self.mode == "near":
        #     dist = np.sqrt((p_x - r_x) ** 2 + (p_y - r_y) ** 2 + (p_z - r_z) ** 2)
        # elif self.mode == "far":
        #     dist = (p_x * r_x) + (p_y * r_y) + (p_z * r_z)

        dist = (p_x * r_x) + (p_y * r_y) + (p_z * r_z)

        tau = dist/self.c
        omega = 2*np.pi*self.sample_rate*np.arange(self.n_fft//2 + 1)/self.n_fft
        mode_vec = np.exp(1j * omega[:, None, None] * tau)

        return mode_vec

    def get_mode_vec(self):
        """
        NOTE:
            mode_vec パラメータ格納時は float なので complex に戻す必要がある
        """
        return torch.view_as_complex(self.mode_vec)
