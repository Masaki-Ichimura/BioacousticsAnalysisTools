"""
    This module is based on Pyroomacoustics (MIT Licence)
        - https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.bss.html

    NOTE:
        n_src, n_iter 以外の引数をデフォルト値から変えたことはないので，
        その辺り適用したらエラー出るかも
"""
import torch
from tqdm import trange

from .base import tf_bss_model_base
from .common import projection_back


class AuxIVA(tf_bss_model_base):
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
        callback=None
    ):

        n_chan, n_freq, n_frames = Xnkl.shape

        X = Xnkl.permute(2, 1, 0)

        # default to determined case
        if n_src is None:
            n_src = n_chan

        assert (
            n_src <= n_chan
        ), "The number of sources cannot be more than the number of channels."

        if model not in ["laplace", "gauss"]:
            raise ValueError("Model should be either " "laplace" " or " "gauss" ".")

        # covariance matrix of input signal (n_freq, n_chan, n_chan)
        Cx = (X[:, :, :, None] * X[:, :, None, :].conj()).mean(0)

        W_hat = torch.zeros(
            (n_freq, n_chan, n_chan), dtype=X.dtype, device=X.device
        )
        W = W_hat[:, :n_src, :]
        J = W_hat[:, n_src:, :n_src]

        def tensor_H(T):
            return T.mT.conj()

        def update_J_from_orth_const():
            tmp = torch.matmul(W, Cx)
            J[:, :, :] = tensor_H(
                torch.linalg.solve(tmp[:, :, :n_src], tmp[:, :, n_src:])
            )

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
                W[:, :, :n_src] = torch.eye(n_src)

        else:
            W[:, :, :] = W0

        # We still need to initialize the rest of the matrix
        if n_src < n_chan:
            update_J_from_orth_const()
            W_hat[:, n_src:, n_src:] = -torch.eye(n_chan - n_src)

        eps = 1e-15
        eyes = torch.eye(n_chan, dtype=X.dtype, device=X.device).tile(n_freq, 1, 1)
        V = torch.zeros((n_freq, n_chan, n_chan), dtype=X.dtype, device=X.device)
        r_inv = torch.zeros((n_src, n_frames), device=X.device)
        r = torch.zeros((n_src, n_frames), device=X.device)

        # Things are more efficient when the frequencies are over the first axis
        Y = torch.zeros((n_freq, n_src, n_frames), dtype=X.dtype, device=X.device)
        X_original = X
        X = X.permute(1, 2, 0).detach().clone()

        # Compute the demixed output
        def demix(Y, X, W):
            Y[:, :, :] = torch.matmul(W, X)

        self.pbar = trange(n_iter)

        for epoch in self.pbar:

            demix(Y, X, W)

            if callback is not None and epoch % 10 == 0:
                Y_tmp = Y.permute(2, 0, 1)
                if proj_back:
                    z = projection_back(Y_tmp, X_original[:, :, 0])
                    callback(Y_tmp * z[None, :, :].conj())
                else:
                    callback(Y_tmp)

            # shape: (n_frames, n_src)
            if model == "laplace":
                r[:, :] = 2.0 * torch.linalg.norm(Y, dim=0)
            elif model == "gauss":
                r[:, :] = torch.linalg.norm(Y, dim=0).square() / n_freq

            # ensure some numerical stability
            r[:] = r.clip(eps)

            r_inv[:, :] = 1.0 / r

            # Update now the demixing matrix
            for s in range(n_src):
                # Compute Auxiliary Variable
                # shape: (n_freq, n_chan, n_chan)
                V[:, :, :] = (
                    torch.matmul(
                        (X * r_inv[None, s, None, :]), X.mT.conj()
                    ) / n_frames
                )

                WV = torch.matmul(W_hat, V)

                W[:, s, :] = torch.linalg.solve(WV, eyes[:, :, s]).conj()

                # normalize
                denom = torch.matmul(
                    torch.matmul(W[:, None, s, :], V[:, :, :]),
                    W[:, s, :, None].conj()
                )
                W[:, s, :] /= torch.sqrt(denom[:, :, 0])

                # Update the mixing matrix according to orthogonal constraints
                if n_src < n_chan:
                    update_J_from_orth_const()

        demix(Y, X, W)

        Y = Y.permute(2, 0, 1).detach().clone()

        if proj_back:
            z = projection_back(Y, X_original[:, :, 0])
            Y *= z[None, :, :].conj()

        Ynkl = Y.permute(2, 1, 0)

        return Ynkl
