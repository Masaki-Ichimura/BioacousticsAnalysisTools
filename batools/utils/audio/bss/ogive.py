"""
    This module is based on overiva
        - License:
            - Apache Licence 2.0
            - https://github.com/onolab-tmu/overiva/blob/master/LICENSE
        - Original @ fakufaku :
            - https://github.com/onolab-tmu/overiva/blob/master/ive.py

        - NOTE:
            - This model returns ONE source
            - This is NOT debugged using GPU tensor yet.
"""

import torch
from tqdm import trange

from .base import tf_bss_model_base
from .common import projection_back


class OGIVE(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        Xnkl,
        n_iter=1000,
        step_size=0.1,
        tol=1e-3,
        update="demix",
        proj_back=True,
        W0=None,
        model="laplace",
        init_eig=False,
        return_filters=False,
        callback=None,
    ):

        n_chan, n_freq, n_frames = Xnkl.shape
        device = Xnkl.device

        X = Xnkl.permute(2, 1, 0)

        n_src = 1

        # covariance matrix of input signal (n_freq, n_chan, n_chan)
        Cx = torch.einsum('lkn,lkm->knm', X, X.conj()) / n_frames
        Cx_inv = torch.linalg.inv(Cx)
        Cx_norm = torch.linalg.norm(Cx, dim=(1, 2))

        w = torch.zeros((n_freq, n_chan, 1), dtype=X.dtype)
        a = torch.zeros((n_freq, n_chan, 1), dtype=X.dtype)
        delta = torch.zeros((n_freq, n_chan, 1), dtype=X.dtype)
        lambda_a = torch.zeros((n_freq, 1, 1), dtype=torch.float64)

        def tensor_H(T):
            return T.mT.conj()

        # eigenvectors of the input covariance
        eigval, eigvec = torch.linalg.eig(Cx)
        lead_eigval = eigval[range(eigval.size(0)), eigval.real.argmax(1)]
        lead_eigvec = torch.zeros((n_freq, n_chan), dtype=Cx.dtype)
        for f in range(n_freq):
            ind = eigval[f].real.argmax()
            lead_eigvec[f, :] = eigvec[f, :, ind]

        # initialize A and W
        if W0 is None:
            if init_eig:

                # Initialize the demixing matrices with the principal
                # eigenvector
                w[:, :, 0] = lead_eigvec

            else:
                # Or with identity
                w[:, 0] = 1.0

        else:
            w[:, :] = W0

        def update_a_from_w(I):
            v_new = Cx[I] @ w[I]
            lambda_w = 1.0 / (tensor_H(w[I]) @ v_new).real
            a[I, :, :] = lambda_w * v_new

        def update_w_from_a(I):
            v_new = Cx_inv @ a
            lambda_a[:] = 1.0 / (tensor_H(a) @ v_new).real
            tmp = lambda_a[I] * v_new[I]
            if tmp.dtype != w.dtype:
                tmp = tmp.type(w.dtype)
            w[I, :, :] = tmp

        def switching_criterion():

            a_n = a / a[:, :1, :1]
            b_n = Cx @ a_n
            lmb = b_n[:, :1, :1].detach().clone()  # copy is important here!
            b_n /= lmb

            p1 = torch.linalg.norm(a_n - b_n, dim=(1, 2)) / Cx_norm
            Cbb = (
                lmb
                * (b_n @ tensor_H(b_n))
                / torch.linalg.norm(b_n, dim=(1, 2), keepdims=True).square()
            )
            p2 = torch.linalg.norm(Cx - Cbb, dim=(1, 2))

            kappa = p1 * p2 / torch.tensor(n_chan).sqrt().item()

            thresh = 0.1
            I_do_a[:] = kappa >= thresh
            I_do_w[:] = kappa < thresh

        # Compute the demixed output
        def demix(Y, X, W):
            Y[:, :, :] = X @ W.conj()

        # The very first update of a
        update_a_from_w(torch.ones(n_freq, dtype=torch.bool))

        if update == "mix":
            I_do_w = torch.zeros(n_freq, dtype=torch.bool)
            I_do_a = torch.ones(n_freq, dtype=torch.bool)
        else:  # default is "demix"
            I_do_w = torch.ones(n_freq, dtype=torch.bool)
            I_do_a = torch.zeros(n_freq, dtype=torch.bool)

        r_inv = torch.zeros((n_frames, n_src))
        r = torch.zeros((n_frames, n_src))

        # Things are more efficient when the frequencies are over the first axis
        Y = torch.zeros((n_freq, n_frames, n_src), dtype=X.dtype)
        X_ref = X  # keep a reference to input signal
        X = X.swapaxes(0, 1).detach().clone()  # more efficient order for processing

        self.pbar = trange(n_iter)

        for epoch in self.pbar:
            # compute the switching criterion
            if update == "switching" and epoch % 10 == 0:
                switching_criterion()

            # Extract the target signal
            demix(Y, X, w)

            # Now run any necessary callback
            if callback is not None and epoch % 100 == 0:
                Y_tmp = Y.swapaxes(0, 1)
                if proj_back:
                    z = projection_back(Y_tmp, X_ref[:, :, 0])
                    callback(Y_tmp * z[None, :, :].conj())
                else:
                    callback(Y_tmp)

            # simple loop as a start
            # shape: (n_frames, n_src)
            if model == "laplace":
                r[:, :] = torch.linalg.norm(Y, axis=0) / torch.tensor(n_freq).sqrt().item()

            elif model == "gauss":
                r[:, :] = (torch.linalg.norm(Y, axis=0) ** 2) / n_freq

            eps = 1e-15
            r[r < eps] = eps

            r_inv[:, :] = 1.0 / r

            # Compute the score function
            psi = r_inv[None, :, :] * Y.conj()

            # "Nu" in Algo 3 in [1]
            # shape (n_freq, 1, 1)
            zeta = Y.swapaxes(1, 2) @ psi

            x_psi = (X.swapaxes(1, 2) @ psi) / zeta

            # The w-step
            # shape (n_freq, n_chan, 1)
            delta[I_do_w] = a[I_do_w] - x_psi[I_do_w]
            w[I_do_w] += step_size * delta[I_do_w]

            # The a-step
            # shape (n_freq, n_chan, 1)
            tmp = w[I_do_a] - (Cx_inv[I_do_a] @ x_psi[I_do_a]) * lambda_a[I_do_a]
            if delta.dtype != tmp.dtype:
                tmp = tmp.type(delta.dtype)
            delta[I_do_a] = tmp
            a[I_do_a] += step_size * delta[I_do_a]

            # Apply the orthogonal constraints
            update_a_from_w(I_do_w)
            update_w_from_a(I_do_a)

            max_delta = torch.linalg.norm(delta, dim=(1, 2)).max()

            if max_delta < tol:
                break

        # Extract target
        demix(Y, X, w)

        Y = Y.swapaxes(0, 1).detach().clone()
        X = X.swapaxes(0, 1)

        if proj_back:
            z = projection_back(Y, X_ref[:, :, 0])
            Y *= z[None, :, :].conj()

        Ynkl = Y.permute(2, 1, 0)

        if return_filters:
            return Ynkl, w
        else:
            return Ynkl
