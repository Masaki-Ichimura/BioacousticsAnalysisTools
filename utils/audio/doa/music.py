"""
    pyroomacousticsからの移植バージョン
        - https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.html
"""
import torch
from .doa import DOA


class MUSIC(DOA):
    def __init__(
        self,
        mic_locs,
        sample_rate,
        n_fft,
        num_src=1,
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
            num_src=num_src,
            r=r,
            n_grid=n_grid,
            mode=mode,
            **kwargs
        )
        self.frequency_normalization = frequency_normalization

    def forward(self, xblkn):
        M = xblkn.shape[-1]

        Cbkmn = self.compute_correlation_matricesvec(xblkn)
        Ebkmn_s, Ebkmn_n, wbkn_s, wbkn_n = self.subspace_decomposition(Cbkmn)

        identity = torch.eye(M, device=xblkn.device)[None, :].tile(self.num_freq, 1, 1)
        Fbkmn = identity-torch.einsum('bkmp,bknp->bkmn', Ebkmn_s, Ebkmn_s.conj())

        Pbrk = self.compute_spatial_spectrumvec(Fbkmn)
        if self.frequency_normalization:
            Pbrk = Pbrk / Pbrk.max(1)[0][:, None, :]

        Pbr = Pbrk.sum(-1)/self.num_freq
        return Pbr

    def compute_correlation_matricesvec(self, xblkn):
        xblkn = xblkn[..., list(self.freq_bins), :]
        Cbkmn = torch.einsum('blkm,blkn->blkmn', xblkn, xblkn.conj()).mean(1)
        return Cbkmn

    def compute_spatial_spectrumvec(self, Rbkmn):
        mode_vec = self.get_mode_vec()
        mrkn = mode_vec[list(self.freq_bins), :, :].permute(2, 0, 1)
        Pbrk = torch.einsum(
            'rkn,brkn->brk',
            mrkn.conj(),
            torch.einsum('bkmn,rkn->brkm', Rbkmn, mrkn)
        ).abs().reciprocal()
        return Pbrk

    def subspace_decomposition(self, Rbkmn):
        w, v = torch.linalg.eigh(Rbkmn)

        Ebkmn_s, Ebkmn_n = v[..., -self.num_src:], v[..., :-self.num_src]
        wbkn_s, wbkn_n = w[..., -self.num_src:], w[..., :-self.num_src]

        return (Ebkmn_s, Ebkmn_n, wbkn_s, wbkn_n)
