"""
    pyroomacousticsからの移植バージョン
        - https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.html
"""
import torch
from .doa import DOA


class SRP(DOA):
    def __init__(
        self,
        mic_locs,
        sample_rate,
        n_fft,
        num_src=1,
        r=1.,
        n_grid=360,
        mode="far",
        tol=1e-14,
        **kwargs
    ):

        super().__init__(
            mic_locs=mic_locs,
            sample_rate=sample_rate,
            n_fft=n_fft,
            num_src=num_src,
            r=r,
            n_grid=n_grid,
            mode=mode,
            **kwargs
        )

        self.tol = tol

        M = self.mic_locs.shape[-1]
        self.num_pairs = M * (M - 1) / 2

    def forward(self, xblkn):
        M = self.mic_locs.shape[-1]

        pblkn = (xblkn/xblkn.abs().clip(self.tol))[..., list(self.freq_bins), :]

        Cbkmn = torch.einsum('blkm,blkn->bkmn', pblkn, pblkn.conj())

        mask_triu = torch.ones(M, M, dtype=bool).triu(diagonal=1)

        C_flat = Cbkmn[..., mask_triu]

        mode_vec = self.get_mode_vec()
        mrkn = mode_vec[list(self.freq_bins), :, :].permute(2, 0, 1)
        mrkmn = torch.einsum('rkm,rkn->rkmn', mrkn.conj(), mrkn)
        m_flat = mrkmn[..., mask_triu]

        Rbrkn = torch.einsum('bkn,rkn->brkn', C_flat.real, m_flat.real) - \
                torch.einsum('bkn,rkn->brkn', C_flat.imag, m_flat.imag)

        offset = xblkn.shape[1] * M * len(self.freq_bins)

        Pbr = (2.0*Rbrkn.sum((-2, -1)) + offset)/(xblkn.shape[1]*self.num_freq*self.num_pairs)
        return Pbr
