"""
    These modules are based on Pyroomacoustics
        - License :
            - MIT License
            - https://github.com/LCAV/pyroomacoustics/blob/pypi-release/LICENSE
"""

import torch
from tqdm import trange

from .base import tf_bss_model_base

EPS =  1e-6
G_EPS = 5e-2


"""
    - Original code @ fakufaku, sekiguchi92 :
        - https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/bss/fastmnmf.py
"""
class FastMNMF(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        Xnkl,
        n_src=None,
        n_iter=30,
        n_components=8,
        mic_index=0,
        W0=None,
        accelerate=True,
        callback=None,
        return_scms=False,
    ):

        interval_update_Q = 1  # 2 may work as well and is faster
        interval_normalize = 10
        TYPE_FLOAT = Xnkl.real.dtype
        TYPE_COMPLEX = Xnkl.dtype

        # initialize parameter
        X_FTM = Xnkl.permute(1, 2, 0)
        n_freq, n_frames, n_chan = X_FTM.size()
        XX_FTMM = torch.einsum('ftm,ftn->ftmn', X_FTM, X_FTM.conj())
        if n_src is None:
            n_src = X_FTM.size(2)

        if W0 is not None:
            Q_FMM = W0
        else:
            Q_FMM = torch.eye(n_chan, dtype=TYPE_COMPLEX).tile(n_freq, 1, 1)

        G_NFM = torch.ones((n_src, n_freq, n_chan), dtype=TYPE_FLOAT) * G_EPS
        for m in range(n_chan):
            G_NFM[m % n_src, :, m] = 1

        for m in range(n_chan):
            mu_F = (Q_FMM[:, m] * Q_FMM[:, m].conj()).sum(1).real
            Q_FMM[:, m] /= mu_F[:, None].sqrt()

        H_NKT = torch.rand((n_src, n_components, n_frames), dtype=TYPE_FLOAT)
        W_NFK = torch.rand((n_src, n_freq, n_components), dtype=TYPE_FLOAT)
        E_FMM = torch.eye(n_src, dtype=TYPE_COMPLEX).tile(n_freq, 1, 1)
        lambda_NFT = W_NFK @ H_NKT + EPS
        Qx_power_FTM = torch.einsum('fij,ftj->fti', Q_FMM, X_FTM).abs()**2
        Y_FTM = torch.einsum('nft,nfm->ftm', lambda_NFT, G_NFM)

        def separate():
            Qx_FTM = torch.einsum('fij,ftj->fti', Q_FMM, X_FTM)
            try:
                Qinv_FMM = Q_FMM.inverse()
            except torch.linalg.LinAlgError:
                # If Gaussian elimination fails due to a singlular matrix, we
                # switch to the pseudo-inverse solution
                import warnings
                warnings.warn("Singular matrix encountered in separate, switching to pseudo-inverse")

                Qinv_FMM = Q_FMM.pinverse()
            Y_NFTM = torch.einsum('nft,nfm->nftm', lambda_NFT, G_NFM)

            if mic_index == "all":
                signals = torch.einsum(
                    'fij,ftj,nftj->itfn', Qinv_FMM, Qx_FTM/Y_NFTM.sum(0), Y_NFTM.type(TYPE_COMPLEX)
                ).permute(3, 0, 2, 1)
            elif type(mic_index) is int:
                signals = torch.einsum(
                    'fj,ftj,nftj->tfn', Qinv_FMM[:, mic_index], Qx_FTM/Y_NFTM.sum(0), Y_NFTM.type(TYPE_COMPLEX)
                ).permute(2, 1, 0)
            else:
                raise ValueError("mic_index should be int or 'all'")

            if return_scms:
                scms = ((X_FTM.mT[None] * lambda_NFT[:, :, None].reciprocal()) @ X_FTM.conj()) / n_frames
                return {'signals': signals, 'scms': scms}
            else:
                return signals

        self.pbar = trange(n_iter)
        # update parameters
        for epoch in self.pbar:
            if callback is not None and epoch % 10 == 0:
                if return_scms:
                    callback(separate()['signals'])
                else:
                    callback(separate())

            # update W and H (basis and activation of NMF)
            tmp1_NFT = torch.einsum('nfm,ftm->nft', G_NFM, Qx_power_FTM/Y_FTM.square())
            tmp2_NFT = torch.einsum('nfm,ftm->nft', G_NFM, Y_FTM.reciprocal())

            numerator = torch.einsum('nkt,nft->nfk', H_NKT, tmp1_NFT)
            denominator = torch.einsum('nkt,nft->nfk', H_NKT, tmp2_NFT) + EPS
            W_NFK *= (numerator / denominator).sqrt()

            if not accelerate:
                lambda_NFT = W_NFK @ H_NKT + EPS
                Y_FTM = torch.einsum('nft,nfm->ftm', lambda_NFT, G_NFM)
                tmp1_NFT = torch.einsum('nfm,ftm->nft', G_NFM, Qx_power_FTM/Y_FTM.square())
                tmp2_NFT = torch.einsum('nfm,ftm->nft', G_NFM, Y_FTM.reciprocal())

            numerator = torch.einsum('nfk,nft->nkt', W_NFK, tmp1_NFT)
            denominator = torch.einsum('nfk,nft->nkt', W_NFK, tmp2_NFT) + EPS
            H_NKT *= (numerator / denominator).sqrt()

            lambda_NFT = W_NFK @ H_NKT + EPS
            Y_FTM = torch.einsum('nft,nfm->ftm', lambda_NFT, G_NFM)

            # update G_NFM (diagonal element of spatial covariance matrices)
            numerator = torch.einsum( 'nft,ftm->nfm', lambda_NFT, Qx_power_FTM/Y_FTM.square())
            denominator = torch.einsum('nft,ftm->nfm', lambda_NFT, Y_FTM.reciprocal()) + EPS
            G_NFM *= (numerator / denominator).sqrt()
            Y_FTM = torch.einsum('nft,nfm->ftm', lambda_NFT, G_NFM)

            # udpate Q (matrix for joint diagonalization)
            if (interval_update_Q <= 0) or (epoch % interval_update_Q == 0):
                for m in range(n_chan):
                    V_FMM = torch.einsum('ftij,ft->fij', XX_FTMM, Y_FTM[..., m].reciprocal().type(TYPE_COMPLEX)) / n_frames
                    QV = Q_FMM @ V_FMM

                    try:
                        tmp_FM = torch.linalg.solve(QV, E_FMM[..., m])
                    except torch.linalg.LinAlgError:
                        # If Gaussian elimination fails due to a singlular matrix, we
                        # switch to the pseudo-inverse solution
                        import warnings
                        warnings.warn("Singular matrix encountered, switching to pseudo-inverse")

                        tmp_FM = (QV.pinverse() @ E_FMM[..., [m]])[..., 0]

                    Q_FMM[:, m] = (
                        tmp_FM / (
                            torch.einsum('fi,fij,fj->f', tmp_FM.conj(), V_FMM, tmp_FM).sqrt()[:, None]
                            + EPS
                        )
                    ).conj()
                    Qx_power_FTM = torch.einsum('fij,ftj->fti', Q_FMM, X_FTM).abs() ** 2

            # normalize
            if (interval_normalize <= 0) or (epoch % interval_normalize == 0):
                phi_F = torch.einsum('fij,fij->f', Q_FMM, Q_FMM.conj()).real / n_chan
                Q_FMM /= phi_F.sqrt()[:, None, None]
                G_NFM /= phi_F[None, :, None]

                mu_NF = G_NFM.sum(2)
                G_NFM /= mu_NF[..., None]
                W_NFK *= mu_NF[..., None]

                nu_NK = W_NFK.sum(1)
                W_NFK /= nu_NK[:, None]
                H_NKT *= nu_NK[:, :, None]

                lambda_NFT = W_NFK @ H_NKT + EPS
                Qx_power_FTM = torch.einsum('fij,ftj->fti', Q_FMM, X_FTM).abs()**2
                Y_FTM = torch.einsum('nft,nfm->ftm', lambda_NFT, G_NFM)

        return separate()

"""
    - Original code @ sekiguchi92 :
        - https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/bss/fastmnmf2.py
"""
class FastMNMF2(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        Xnkl,
        n_src=None,
        n_iter=30,
        n_components=8,
        mic_index=0,
        W0=None,
        accelerate=True,
        callback=None,
        return_scms=False,
    ):
        interval_update_Q = 1  # 2 may work as well and is faster
        interval_normalize = 10
        TYPE_FLOAT = Xnkl.real.dtype
        TYPE_COMPLEX = Xnkl.dtype

        # initialize parameter
        X_FTM = Xnkl.permute(1, 2, 0)
        n_freq, n_frames, n_chan = X_FTM.size()
        XX_FTMM = torch.einsum('ftm,ftn->ftmn', X_FTM, X_FTM.conj())
        if n_src is None:
            n_src = X_FTM.size(2)

        if W0 is not None:
            Q_FMM = W0
        else:
            Q_FMM = torch.eye(n_chan, dtype=TYPE_COMPLEX).tile(n_freq, 1, 1)

        g_NM = torch.ones((n_src, n_chan), dtype=TYPE_FLOAT) * G_EPS
        for m in range(n_chan):
            g_NM[m % n_src, m] = 1

        for m in range(n_chan):
            mu_F = (Q_FMM[:, m] * Q_FMM[:, m].conj()).sum(1).real
            Q_FMM[:, m] /= mu_F[:, None].sqrt()

        H_NKT = torch.rand((n_src, n_components, n_frames), dtype=TYPE_FLOAT)
        W_NFK = torch.rand((n_src, n_freq, n_components), dtype=TYPE_FLOAT)
        E_FMM = torch.eye(n_src, dtype=TYPE_COMPLEX).tile(n_freq, 1, 1)
        lambda_NFT = W_NFK @ H_NKT
        Qx_power_FTM = torch.einsum('fij,ftj->fti', Q_FMM, X_FTM).abs()**2
        Y_FTM = torch.einsum('nft,nm->ftm', lambda_NFT, g_NM)

        def separate():
            Qx_FTM = torch.einsum('fij,ftj->fti', Q_FMM, X_FTM)
            try:
                Qinv_FMM = Q_FMM.inverse()
            except torch.linalg.LinAlgError:
                # If Gaussian elimination fails due to a singlular matrix, we
                # switch to the pseudo-inverse solution
                import warnings
                warnings.warn("Singular matrix encountered in separate, switching to pseudo-inverse")

                Qinv_FMM = Q_FMM.pinverse()
            Y_NFTM = torch.einsum('nft,nm->nftm', lambda_NFT, g_NM)

            if mic_index == "all":
                signals = torch.einsum(
                    'fij,ftj,nftj->itfn', Qinv_FMM, Qx_FTM/Y_NFTM.sum(0), Y_NFTM.type(TYPE_COMPLEX)
                ).permute(3, 0, 2, 1)
            elif type(mic_index) is int:
                signals = torch.einsum(
                    'fj,ftj,nftj->tfn', Qinv_FMM[:, mic_index], Qx_FTM/Y_NFTM.sum(0), Y_NFTM.type(TYPE_COMPLEX)
                ).permute(2, 1, 0)
            else:
                raise ValueError("mic_index should be int or 'all'")

            if return_scms:
                # (X:imj->1imj,rR:nij->ni1j)->nimj,XmT:ijm=>H:nimm
                scms = ((X_FTM.mT[None] * lambda_NFT[:, :, None].reciprocal()) @ X_FTM.conj()) / n_frames
                return {'signals': signals, 'scms': scms}
            else:
                return signals

        self.pbar = trange(n_iter)

        # update parameters
        for epoch in self.pbar:
            if callback is not None and epoch % 10 == 0:
                if return_scms:
                    callback(separate()['signals'])
                else:
                    callback(separate())

            # update W and H (basis and activation of NMF)
            tmp1_NFT = torch.einsum('nm,ftm->nft', g_NM, Qx_power_FTM/Y_FTM.square())
            tmp2_NFT = torch.einsum('nm,ftm->nft', g_NM, Y_FTM.reciprocal())

            numerator = torch.einsum('nkt,nft->nfk', H_NKT, tmp1_NFT)
            denominator = torch.einsum('nkt,nft->nfk', H_NKT, tmp2_NFT)
            W_NFK *= (numerator / denominator).sqrt()

            if not accelerate:
                tmp1_NFT = torch.einsum('nm,ftm->nft', g_NM, Qx_power_FTM/Y_FTM.square())
                tmp2_NFT = torch.einsum('nm,ftm->nft', g_NM, Y_FTM.reciprocal())
                lambda_NFT = W_NFK @ H_NKT + EPS
                Y_FTM = torch.einsum('nft,nm->ftm', lambda_NFT, g_NM) + EPS

            numerator = torch.einsum('nfk,nft->nkt', W_NFK, tmp1_NFT)
            denominator = torch.einsum('nfk,nft->nkt', W_NFK, tmp2_NFT)
            H_NKT *= (numerator / denominator).sqrt()

            lambda_NFT = W_NFK @ H_NKT + EPS
            Y_FTM = torch.einsum('nft,nm->ftm', lambda_NFT, g_NM) + EPS

            # update g_NM (diagonal element of spatial covariance matrices)
            numerator = torch.einsum('nft,ftm->nm', lambda_NFT, Qx_power_FTM/Y_FTM.square())
            denominator = torch.einsum('nft,ftm->nm', lambda_NFT, Y_FTM.reciprocal())
            g_NM *= (numerator / denominator).sqrt()
            Y_FTM = torch.einsum('nft,nm->ftm', lambda_NFT, g_NM) + EPS

            # udpate Q (joint diagonalizer)
            if (interval_update_Q <= 0) or (epoch % interval_update_Q == 0):
                for m in range(n_chan):
                    V_FMM = torch.einsum('ftij,ft->fij', XX_FTMM, Y_FTM[..., m].reciprocal().type(TYPE_COMPLEX)) / n_frames
                    QV = Q_FMM @ V_FMM

                    try:
                        tmp_FM = torch.linalg.solve(QV, E_FMM[..., m])
                    except torch.linalg.LinAlgError:
                        # If Gaussian elimination fails due to a singlular matrix, we
                        # switch to the pseudo-inverse solution
                        import warnings
                        warnings.warn("Singular matrix encountered, switching to pseudo-inverse")

                        tmp_FM = (QV.pinverse() @ E_FMM[..., [m]])[..., 0]

                    Q_FMM[:, m] = (
                        tmp_FM / (
                            torch.einsum('fi,fij,fj->f', tmp_FM.conj(), V_FMM, tmp_FM).sqrt()[:, None] + EPS
                        )
                    ).conj()
                    Qx_power_FTM = torch.einsum('fij,ftj->fti', Q_FMM, X_FTM).abs() ** 2

            # normalize
            if (interval_normalize <= 0) or (epoch % interval_normalize == 0):
                phi_F = torch.einsum('fij,fij->f', Q_FMM, Q_FMM.conj()).real / n_chan
                Q_FMM /= phi_F.sqrt()[:, None, None]
                W_NFK /= phi_F[None, :, None]

                mu_N = g_NM.sum(1)
                g_NM /= mu_N[:, None]
                W_NFK *= mu_N[:, None, None]

                nu_NK = W_NFK.sum(1)
                W_NFK /= nu_NK[:, None]
                H_NKT *= nu_NK[:, :, None]

                lambda_NFT = W_NFK @ H_NKT + EPS
                Qx_power_FTM = torch.einsum('fij,ftj->fti', Q_FMM, X_FTM).abs()**2
                Y_FTM = torch.einsum('nft,nm->ftm', lambda_NFT, g_NM) + EPS

        return separate()
