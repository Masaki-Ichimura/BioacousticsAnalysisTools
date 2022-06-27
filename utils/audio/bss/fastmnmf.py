"""
    This module is based on Pyroomacoustics (MIT Licence)
        - https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.bss.html
"""
import torch
from tqdm import trange

from .base import tf_bss_model_base


class FastMNMF(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        Xnkl,
        n_src=None,
        n_iter=30,
        W0=None,
        n_components=4,
        callback=None,
        mic_index=0,
        interval_update_Q=1,
        interval_normalize=10,
        initialize_ilrma=False,
    ):

        eps = 1e-7

        n_chan, n_freq, n_frames = Xnkl.shape

        # initialize parameter
        X_FTM = Xnkl.permute(1, 2, 0)
        dtype, device = Xnkl.dtype, Xnkl.device

        XX_FTMM = torch.matmul(X_FTM[:, :, :, None], X_FTM[:, :, None, :].conj())
        if n_src is None:
            n_src = X_FTM.shape[
                2
            ]  # determined case (the number of source = the number of microphone)

        if initialize_ilrma:  # initialize by using ILRMA
            from .utils.bss.ilrma import ILRMA

            ilrma = ILRMA(**self.stft_args).to(device)

            Y_TFM, W = ilrma.separate(
                X, n_iter=10, n_components=2, proj_back=False, return_filters=True
            )
            Q_FMM = W
            sep_power_M = Y_TFM.abs().mean(dim=(0, 1))
            g_NFM = torch.ones((n_src, n_freq, n_chan), device=device) * 1e-2
            for n in range(n_src):
                g_NFM[n, :, sep_power_M.argmax()] = 1
                sep_power_M[sep_power_M.argmax()] = 0
        elif W0 != None:  # initialize by W0
            Q_FMM = W0
            g_NFM = torch.ones((n_src, n_freq, n_chan), device=device) * 1e-2
            for m in range(n_chan):
                g_NFM[m % n_src, :, m] = 1
        else:  # initialize by using observed signals
            Q_FMM = torch.tile(
                torch.eye(n_chan, dtype=dtype, device=device), (n_freq, 1, 1)
            )
            g_NFM = torch.ones((n_src, n_freq, n_chan), device=device) * 1e-2
            for m in range(n_chan):
                g_NFM[m % n_src, :, m] = 1

        for m in range(n_chan):
            mu_F = (Q_FMM[:, m] * Q_FMM[:, m].conj()).real.sum(dim=1)
            Q_FMM[:, m] = Q_FMM[:, m] / torch.sqrt(mu_F[:, None])
        H_NKT = torch.rand(n_src, n_components, n_frames, device=device)
        W_NFK = torch.rand(n_src, n_freq, n_components, device=device)
        lambda_NFT = torch.matmul(W_NFK, H_NKT)
        Qx_power_FTM = (
            torch.matmul(Q_FMM[:, None], X_FTM[:, :, :, None])[:, :, :, 0].abs() ** 2
        )
        Y_FTM = (lambda_NFT[..., None] * g_NFM[:, :, None]).sum(dim=0)

        separated_spec = torch.zeros([n_src, n_freq, n_frames], device=device, dtype=dtype)

        def separate():
            Qx_FTM = (Q_FMM[:, None] * X_FTM[:, :, None]).sum(dim=3)
            diagonalizer_inv_FMM = torch.linalg.inv(Q_FMM)
            tmp_NFTM = lambda_NFT[..., None] * g_NFM[:, :, None]
            for n in range(n_src):
                tmp = (
                    torch.matmul(
                        diagonalizer_inv_FMM[:, None],
                        (Qx_FTM * (tmp_NFTM[n] / (tmp_NFTM).sum(dim=0)))[..., None],
                    )
                )[:, :, mic_index, 0]
                separated_spec[n] = tmp
            return separated_spec

        # update parameters
        for epoch in trange(n_iter):
            if callback is not None and epoch % 10 == 0:
                callback(separate())

            # update_WH (basis and activation of NMF)
            tmp_yb1 = (
                g_NFM[:, :, None] * (Qx_power_FTM / (Y_FTM ** 2))[None]
            ).sum(dim=3) # [N, F, T]
            tmp_yb2 = (g_NFM[:, :, None] / Y_FTM[None]).sum(dim=3)  # [N, F, T]
            a_1 = (H_NKT[:, None, :, :] * tmp_yb1[:, :, None]).sum(dim=3)  # [N, F, K]
            b_1 = (H_NKT[:, None, :, :] * tmp_yb2[:, :, None]).sum(dim=3)  # [N, F, K]
            W_NFK *= torch.sqrt(a_1 / b_1)

            a_1 = (W_NFK[:, :, :, None] * tmp_yb1[:, :, None]).sum(dim=1)  # [N, K, T]
            b_1 = (W_NFK[:, :, :, None] * tmp_yb2[:, :, None]).sum(dim=1)  # [N, F, K]
            H_NKT *= torch.sqrt(a_1 / b_1)

            lambda_NFT = torch.matmul(W_NFK, H_NKT)
            lambda_NFT = lambda_NFT.clip(eps)
            Y_FTM = (lambda_NFT[..., None] * g_NFM[:, :, None]).sum(dim=0)

            # update diagonal element of spatial covariance matrix
            a_1 = (
                lambda_NFT[..., None] * (Qx_power_FTM / (Y_FTM ** 2))[None]
            ).sum(dim=2) # N F T M
            b_1 = (lambda_NFT[..., None] / Y_FTM[None]).sum(dim=2)
            g_NFM *= torch.sqrt(a_1 / b_1)
            Y_FTM = (lambda_NFT[..., None] * g_NFM[:, :, None]).sum(dim=0)

            # udpate Diagonalizer which jointly diagonalize spatial covariance matrix
            if (interval_update_Q <= 0) or ((epoch + 1) % interval_update_Q == 0):
                for m in range(n_chan):
                    V_FMM = (XX_FTMM / Y_FTM[:, :, m, None, None]).mean(dim=1)
                    tmp_FM = torch.linalg.solve(
                        torch.matmul(Q_FMM, V_FMM),
                        torch.eye(n_chan, dtype=dtype, device=device)[m]
                    )
                    Q_FMM[:, m] = (
                        tmp_FM
                        / torch.sqrt(
                            (
                                (tmp_FM.conj()[:, :, None] * V_FMM).sum(dim=1)*tmp_FM
                            ).sum(dim=1)
                        )[:, None]
                    ).conj()
                Qx_power_FTM = (
                    torch.matmul(
                        Q_FMM[:, None], X_FTM[:, :, :, None]
                    )[:, :, :, 0].abs() ** 2
                )

            # normalize
            if (interval_normalize <= 0) or (epoch % interval_normalize == 0):
                phi_F = torch.sum(Q_FMM * Q_FMM.conj(), dim=(1, 2)).real / n_chan
                Q_FMM /= torch.sqrt(phi_F)[:, None, None]
                g_NFM /= phi_F[None, :, None]

                mu_NF = (g_NFM).sum(dim=2)
                g_NFM /= mu_NF[:, :, None]
                W_NFK *= mu_NF[:, :, None]

                mu_NK = W_NFK.sum(dim=1)
                W_NFK /= mu_NK[:, None]
                H_NKT *= mu_NK[:, :, None]
                lambda_NFT = torch.matmul(W_NFK, H_NKT)
                lambda_NFT = lambda_NFT.clip(1e-10)

                Qx_power_FTM = (
                    torch.matmul(
                        Q_FMM[:, None], X_FTM[:, :, :, None]
                    )[:, :, :, 0].abs() ** 2
                )
                Y_FTM = (lambda_NFT[..., None] * g_NFM[:, :, None]).sum(dim=0)

        return separate()
