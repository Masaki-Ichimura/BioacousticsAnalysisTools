"""
    This module is based on Pyroomacoustics
        - License:
            - MIT License
            - https://github.com/LCAV/pyroomacoustics/blob/pypi-release/LICENSE
        - Original @ fakufaku, jazcarretao, mori97 :
            - https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/bss/ilrma.py
"""

import torch
from tqdm import trange

from .base import tf_bss_model_base
from .common import projection_back


class ILRMA(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        Xnkl,
        n_src=None,
        n_iter=20,
        proj_back=True,
        W0=None,
        n_components=2,
        return_filters=False,
        callback=None,
    ):
        n_chan, n_freq, n_frames = Xnkl.shape
        dtype, device = Xnkl.dtype, Xnkl.device

        X = Xnkl.permute(2, 1, 0)

        # default to determined case
        if n_src is None:
            n_src = n_chan

        # Only supports determined case
        assert n_chan == n_src, "There should be as many microphones as sources"

        # initialize the demixing matrices
        # The demixing matrix has the following dimensions (nfrequencies, nchannels, nsources),
        if W0 is None:
            W = torch.eye(n_chan, n_src, dtype=dtype, device=device).tile(n_freq, 1, 1)
        else:
            W = W0.detach().clone()

        # initialize the nonnegative matrixes with random values
        T = 0.1 + 0.9*torch.rand((n_src, n_freq, n_components), device=device)
        V = 0.1 + 0.9*torch.rand((n_src, n_frames, n_components), device=device)
        R = torch.zeros((n_src, n_freq, n_frames), device=device)
        I = torch.eye(n_src, dtype=dtype, device=device)
        # U = torch.zeros((n_freq, n_src, n_chan, n_chan), dtype=dtype, device=device)
        # product = torch.zeros((n_freq, n_chan, n_chan), dtype=dtype)
        lambda_aux = torch.zeros(n_src)
        eps = 1e-15
        eyes = I.tile(n_freq, 1, 1)

        # Things are more efficient when the frequencies are over the first axis
        Y = torch.zeros((n_freq, n_src, n_frames), dtype=dtype, device=device)
        X_original = X
        X = X.permute(1, 2, 0)

        torch.matmul(T, V.mT, out=R)

        # Compute the demixed output
        def demix(Y, X, W):
            torch.matmul(W, X, out=Y)

        demix(Y, X, W)

        # P.shape == R.shape == (n_src, n_freq, n_frames)
        P = (Y*Y.conj()).real.permute(1, 0, 2)
        iR = (1 / R).type(P.dtype)

        if T.dtype!=P.dtype or V.dtype!=P.dtype:
            T, V = T.type(P.dtype), V.type(P.dtype)

        self.pbar = trange(n_iter)

        for epoch in self.pbar:
            if callback is not None and epoch % 10 == 0:
                Y_t = Y.permute(2, 0, 1)
                if proj_back:
                    z = projection_back(Y_t, X_original[:, :, 0])
                    callback(Y_t * z[None].conj())
                else:
                    callback(Y_t)

            # simple loop as a start
            for s in range(n_src):
                ## NMF
                ######

                T[s, :, :] *= (((P[s] * iR[s]**2) @ V[s]) / (iR[s] @ V[s])).sqrt()
                T[:] = T.clamp(min=eps)

                R[s, :, :] = (T[s] @ V[s].T).clamp(min=eps)

                iR[s, :, :] = 1 / R[s]

                V[s, :, :] *= (((P[s].T * iR[s].T**2) @ T[s]) / (iR[s].T @ T[s])).sqrt()
                V[:] = V.clamp(min=eps)

                R[s, :, :] = (T[s] @ V[s].T).clamp(min=eps)

                iR[s, :, :] = 1 / R[s]

                ## IVA
                ######

                # Compute Auxiliary Variable
                # shape: (n_freq, n_chan, n_chan)
                C = ((X * iR[s, :, None, :]) @ X.mT.conj()) / n_frames

                WV = W @ C
                try:
                    W[:, s, :] = torch.linalg.solve(WV, eyes[..., s]).conj()
                except torch.linalg.LinAlgError:
                    import warnings

                    warnings.warn("Singular matrix encountered, switching to pseudo-inverse")

                    W[:, s, :] = (WV.pinverse() @ eyes[..., [s]])[..., 0]

                # normalize
                denom = (W[:, None, s] @ C) @ W[:, s, :, None].conj()
                W[:, s, :] /= denom[..., 0].sqrt()

            demix(Y, X, W)
            P[:] = (Y*Y.conj()).real.permute(1, 0, 2)

            for s in range(n_src):
                lambda_aux[s] = 1 / P[s].mean().sqrt()

                W[:, :, s] *= lambda_aux[s]
                P[s, :, :] *= lambda_aux[s] ** 2
                R[s, :, :] *= lambda_aux[s] ** 2
                T[s, :, :] *= lambda_aux[s] ** 2

        Y = Y.permute(2, 0, 1).detach().clone()

        if proj_back:
            z = projection_back(Y, X_original[..., 0])
            Y *= z[None].conj()

        Ynkl = Y.permute(2, 1, 0)

        if return_filters:
            return Ynkl, W
        else:
            return Ynkl
