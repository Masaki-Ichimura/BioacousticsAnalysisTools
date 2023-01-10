"""
    This module is based on Pyroomacoustics
        - License :
            - MIT License
            - https://github.com/LCAV/pyroomacoustics/blob/pypi-release/LICENSE
        - Original @ fakufaku, 4bian, Womac :
            - https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/doa/music.py
"""

import torch
from .doa import DOA


class MUSIC(DOA):
    def __init__(
        self,
        mic_locs,
        sample_rate,
        n_fft,
        n_src=1,
        r=1.,
        n_grid=360,
        mode="far",
        frequency_normalization=False,
        **kwargs
    ):
        super().__init__(
            mic_locs=mic_locs,
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_src=n_src,
            r=r,
            n_grid=n_grid,
            mode=mode,
            **kwargs
        )
        self.frequency_normalization = frequency_normalization

    def forward(self, signals=None, scms=None):
        assert signals is not None or scms is not None, 'DoA estimation needs signals or spacial correlation matrices.'

        if signals is not None:
            Xblkm = signals
            Cbkmm = self.compute_correlation_matricesvec(Xblkm)
        elif scms is not None:
            Cbkmm = scms[:, list(self.freq_bins)]

        M = Cbkmm.size(-1)

        Ebkmm_s, Ebkmm_n, wbkm_s, wbkm_n = self.subspace_decomposition(Cbkmm)

        identity = torch.eye(M, device=Cbkmm.device).tile(self.num_freq, 1, 1)
        Fbkmm = identity-torch.einsum('bkmp,bknp->bkmn', Ebkmm_s, Ebkmm_s.conj())

        Pbrk = self.compute_spatial_spectrumvec(Fbkmm)
        if self.frequency_normalization:
            Pbrk = Pbrk / Pbrk.max(1, keepdim=True).values

        Pbr = Pbrk.mean(-1)
        return Pbr

    def compute_correlation_matricesvec(self, Xblkn):
        Xblkn = Xblkn[..., list(self.freq_bins), :]
        Cbkmm = torch.einsum('blkm,blkn->blkmn', Xblkn, Xblkn.conj()).mean(1)
        return Cbkmm

    def compute_spatial_spectrumvec(self, Rbkmm):
        mode_vec = self.get_mode_vec()
        mrkm = mode_vec[list(self.freq_bins)].permute(2, 0, 1)
        Pbrk = torch.einsum(
            'rkn,brkn->brk', mrkm.conj(), torch.einsum('bkmn,rkn->brkm', Rbkmm, mrkm)
        ).abs().reciprocal()
        return Pbrk

    def subspace_decomposition(self, Rbkmm):
        w, v = torch.linalg.eigh(Rbkmm)
        indices = w.real.argsort(dim=-1)
        w = torch.take_along_dim(w, indices, dim=-1)
        v = torch.take_along_dim(v, indices[..., None, :], dim=-1)

        Ebkmm_s, Ebkmm_n = v[..., -self.n_src:], v[..., :-self.n_src]
        wbkm_s, wbkm_n = w[..., -self.n_src:], w[..., :-self.n_src]

        return (Ebkmm_s, Ebkmm_n, wbkm_s, wbkm_n)
