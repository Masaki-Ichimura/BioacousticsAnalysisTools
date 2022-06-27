"""
    This module is based on overiva by fakufaku (MIT Licence)
        - https://github.com/onolab-tmu/overiva/blob/master/overiva.py

    NOTE:
        - Over determined IVA
        - This is NOT debugged using GPU tensor yet.
"""

import torch
from tqdm import trange

from .base import tf_bss_model_base
from .common import projection_back


class OverIVA(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        Xnkl,
        n_src=None,
        n_iter=20,
        proj_back=True,
        W0=None,
        model="laplace",
        init_eig=False,
        return_filters=False,
        callback=None
    ):

        n_chan, n_freq, n_frames = Xnkl.shape
        device = Xnkl.device

        X = Xnkl.permute(2, 1, 0)

        # default to determined case
        if n_src is None:
            n_src = n_chan

        if model not in ["laplace", "gauss"]:
            raise ValueError("Model should be either " "laplace" " or " "gauss" ".")

        # covariance matrix of input signal (n_freq, n_chan, n_chan)
        Cx = torch.einsum('lkn,lkm->knm', X, X.conj()) / n_frames

        W_hat = torch.zeros((n_freq, n_chan, n_chan), dtype=X.dtype, device=device)
        W = W_hat[:, :, :n_src]
        J = W_hat[:, :n_src, n_src:]

        def tensor_H(T):
            return T.mT.conj()

        def update_J_from_orth_const():
            tmp = torch.matmul(tensor_H(W), Cx)
            J[:, :, :] = torch.linalg.solve(tmp[:, :, :n_src], tmp[:, :, n_src:])

        # initialize A and W
        if W0 is None:

            if init_eig:
                # Initialize the demixing matrices with the principal
                # eigenvectors of the input covariance
                v, w = torch.linalg.eig(Cx)
                for f in range(n_freq):
                    ind = torch.argsort(v[f])[-n_src:]
                    W[f, :, :] = w[f][:, ind].conj()

            else:
                # Or with identity
                for f in range(n_freq):
                    W[f, :n_src, :] = torch.eye(n_src)

        else:
            W[:, :, :] = W0

        # We still need to initialize the rest of the matrix
        if n_src < n_chan:
            update_J_from_orth_const()
            for f in range(n_freq):
                W_hat[f, n_src:, n_src:] = -torch.eye(n_chan - n_src)

        eyes = torch.eye(n_chan, dtype=X.dtype, device=device).tile(n_freq, 1, 1)
        V = torch.zeros((n_freq, n_chan, n_chan), dtype=X.dtype, device=device)
        r_inv = torch.zeros((n_frames, n_src), device=device)
        r = torch.zeros((n_frames, n_src), device=device)

        # Things are more efficient when the frequencies are over the first axis
        Y = torch.zeros((n_freq, n_frames, n_src), dtype=X.dtype, device=device)
        X = X.swapaxes(0, 1).detach().clone()

        # Compute the demixed output
        def demix(Y, X, W):
            Y[:, :, :] = X @ W.conj()

        for epoch in trange(n_iter):

            demix(Y, X, W)

            if callback is not None and epoch % 10 == 0:
                Y_tmp = Y.swapaxes(0, 1)
                if proj_back:
                    z = projection_back(Y_tmp, X[:, :, 0].swapaxes(0, 1))
                    callback(Y_tmp * z[None, :, :].conj())
                else:
                    callback(Y_tmp)

            # simple loop as a start
            # shape: (n_frames, n_src)
            if model == 'laplace':
                r[:, :] = (2. * torch.linalg.norm(Y, dim=0))
            elif model == 'gauss':
                r[:, :] = torch.linalg.norm(Y, dim=0).square() / n_freq

            # set the scale of r
            gamma = r.mean(0)
            r /= gamma[None, :]

            if model == 'laplace':
                Y /= gamma[None, None, :]
                W /= gamma[None, None, :]
            elif model == 'gauss':
                g_sq = gamma[None, None, :].sqrt()
                Y /= g_sq
                W /= g_sq

            # ensure some numerical stability
            eps = 1e-15
            r.clip_(min=eps)

            r_inv[:, :] = 1. / r

            # Update now the demixing matrix
            for s in range(n_src):
                # Compute Auxiliary Variable
                # shape: (n_freq, n_chan, n_chan)
                V[:, :, :] = (X.swapaxes(1, 2) * r_inv[None, None, :, s]) @ X.conj() / n_frames

                WV = W_hat.conj().swapaxes(1, 2) @ V
                W[:, :, s] = torch.linalg.solve(WV, eyes[:, :, s])

                # normalize
                denom = W[:, None, :, s].conj() @ V[:, :, :] @ W[:, :, None, s]
                W[:, :, s] /= denom[:, :, 0].sqrt()

                # Update the mixing matrix according to orthogonal constraints
                if n_src < n_chan:
                    update_J_from_orth_const()

        demix(Y, X, W)

        Y = Y.swapaxes(0, 1).detach().clone()
        X = X.swapaxes(0, 1)

        if proj_back:
            z = projection_back(Y, X[:, :, 0])
            Y *= z[None, :, :].conj()

        Ynkl = Y.permute(2, 1, 0)

        if return_filters:
            return Ynkl, W
        else:
            return Ynkl
