import torch
from tqdm import trange

from .base import tf_bss_model_base
from .common import projection_back

EPS = 1e-15


"""
    ILRMA is based on Pyroomacoustics
        - License :
            - MIT License
            - https://github.com/LCAV/pyroomacoustics/blob/pypi-release/LICENSE
        - Original code @ fakufaku, jazcarretao, mori97 :
            - https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/bss/ilrma.py
"""
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
        return_scms=False,
        callback=None,
    ):
        n_chan, n_freq, n_frames = Xnkl.shape
        dtype, device = Xnkl.dtype, Xnkl.device

        X = Xnkl.permute(1, 0, 2)

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
            W = W0

        # Things are more efficient when the frequencies are over the first axis
        Y = torch.zeros((n_freq, n_src, n_frames), dtype=dtype, device=device)

        # Compute the demixed output
        def demix(Y, X, W):
            Y[:] = torch.einsum('inm,imj->inj', W, X)

        demix(Y, X, W)

        # initialize the nonnegative matrixes with random values
        P = torch.einsum('inj,inj->nij', Y, Y.conj()).real.clamp(min=EPS)
        T = 0.1 + 0.9*torch.rand((n_src, n_freq, n_components), device=device)
        V = 0.1 + 0.9*torch.rand((n_src, n_frames, n_components), device=device)
        R = T @ V.mT
        rR = R.reciprocal()
        E = torch.eye(n_src, dtype=dtype, device=device).tile(n_freq, 1, 1)

        lambda_aux = torch.zeros(n_src)

        self.pbar = trange(n_iter)
        for epoch in self.pbar:
            if callback is not None and epoch % 10 == 0:
                Y_t = Y.permute(2, 0, 1)
                if proj_back:
                    z = projection_back(Y_t, X[:, 0].T)
                    callback(Y_t * z[None].conj())
                else:
                    callback(Y_t)

            # if epoch < n_iter//4:
            #     n_perm = torch.stack([torch.randperm(n_src) for i in range(n_freq)])
            #     i_perm = torch.arange(n_freq)[:, None]
            #     W[:] = W[i_perm, n_perm, :]
            #     P[:] = P.permute(1, 0, 2)[i_perm, n_perm, :].permute(1, 0, 2)
            #     R[:] = R.permute(1, 0, 2)[i_perm, n_perm, :].permute(1, 0, 2)
            #     rR[:] = rR.permute(1, 0, 2)[i_perm, n_perm, :].permute(1, 0, 2)

            # simple loop as a start
            for n in range(n_src):
                ## NMF
                ######
                T[n, :, :] *= (((P[n] * rR[n]**2) @ V[n]) / (rR[n] @ V[n])).sqrt()
                T[n, :, :] = T[n].clamp(min=EPS)
                R[n, :, :] = (T[n] @ V[n].T).clamp(min=EPS)
                rR[n, :, :] = R[n].reciprocal()

                V[n, :, :] *= (((P[n].T * rR[n].T**2) @ T[n]) / (rR[n].T @ T[n])).sqrt()
                V[n, :, :] = V[n].clamp(min=EPS)
                R[n, :, :] = (T[n] @ V[n].T).clamp(min=EPS)
                rR[n, :, :] = R[n].reciprocal()

                ## IVA
                ######

                # Compute Auxiliary Variable
                # shape: (n_freq, n_chan, n_chan)
                C = ((X * rR[n, :, None, :]) @ X.mT.conj()) / n_frames

                WV = W @ C
                try:
                    W[:, n, :] = torch.linalg.solve(WV, E[..., n]).conj()
                except torch.linalg.LinAlgError:
                    import warnings
                    warnings.warn("Singular matrix encountered, switching to pseudo-inverse")

                    W[:, n, :] = (WV.pinverse() @ E[..., [n]])[..., 0]

                # normalize
                denom = (W[:, [n]] @ C) @ W[:, n, :, None].conj()
                W[:, n, :] /= denom[..., 0].sqrt()

            demix(Y, X, W)

            P[:] = torch.einsum('inj,inj->nij', Y, Y.conj()).real.clamp(min=EPS)

            for n in range(n_src):
                lambda_aux[n] = P[n].mean().reciprocal()

                W[:, n, :] *= lambda_aux[n].sqrt()
                P[n, :, :] *= lambda_aux[n]
                R[n, :, :] *= lambda_aux[n]
                T[n, :, :] *= lambda_aux[n]

        if proj_back:
            z = projection_back(Y.permute(2, 0, 1), X[:, 0].T)
            Y *= z[..., None].conj()

        ret_val = {'signals': Y.permute(1, 0, 2)}

        if return_filters:
            ret_val['filters'] = W

        if return_scms:
            # (X:imj->1imj,rR:nij->ni1j)->nimj,XmT:ijm=>H:nimm
            H = ((X[None] * rR[:, :, None]) @ X.mT.conj()) / n_frames
            ret_val['scms'] = H

        return ret_val

"""
    - Original paper :
        - D. Kitamura and K. Yatabe, "Consistent independent low-rank matrix analysis
        for determined blind source separation," EURASIP J. Adv. Signal Process.,
        vol. 2020, no. 46, p. 35, November 2020.

    - Original code @ d-kitamura :
        - https://github.com/d-kitamura/ILRMA/blob/master/consistentILRMA.m
"""
class ConsistentILRMA(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def calc_cost_fn(self, P, R, W):
        n_frames = P.size(2)

        log_det_abs = W.det().abs().clamp(min=EPS).log()
        cost = (P/R + R.log()).sum() - 2*n_frames*log_det_abs.sum()

        return cost

    def separate(
        self,
        Xnkl,
        n_src=None,
        n_iter=20,
        n_components=2,
        W0=None,
        proj_back=True,
        return_filters=False,
        return_scms=False,
    ):
        n_chan, n_freq, n_frames = Xnkl.shape
        dtype, device = Xnkl.dtype, Xnkl.device

        if n_src is None:
            n_src = n_chan

        assert n_chan == n_src, "There should be as many microphones as sources"

        X = Xnkl.permute(1, 0, 2)

        if W0 is None:
            W = torch.eye(n_src, dtype=dtype, device=device).tile(n_freq, 1, 1)
        else:
            W = W0

        Y = torch.zeros((n_freq, n_src, n_frames), dtype=dtype, device=device)

        def demix(W, X, Y):
            Y[:] = torch.einsum('inm,imj->inj', W, X)

        demix(W, X, Y)

        P = torch.einsum('inj,inj->nij', Y, Y.conj()).real.clamp(min=EPS)
        T = torch.rand((n_src, n_freq, n_components), device=device).clamp(min=EPS)
        V = torch.rand((n_src, n_components, n_frames), device=device).clamp(min=EPS)
        R = T @ V
        rR = R.reciprocal()
        E = torch.eye(n_src, dtype=dtype, device=device).tile(n_freq, 1, 1)
        # cost = torch.zeros(n_iter+1)

        # cost[0] = self.calc_cost_fn(P, R, W)

        self.pbar = trange(n_iter)
        for epoch in self.pbar:
            for n in range(n_src):
                T[n, :, :] *= (((P[n]*rR[n].square()) @ V[n].T) / (rR[n] @ V[n].T)).sqrt()
                T[n, :, :] = T[n].clamp(min=EPS)
                R[n, :, :] = T[n] @ V[n]
                rR[n, :, :] = R[n].reciprocal()

                V[n, :, :] *= ((T[n].T @ (P[n]*rR[n].square())) / (T[n].T @ rR[n])).sqrt()
                V[n, :, :] = V[n].clamp(min=EPS)
                R[n, :, :] = T[n] @ V[n]
                rR[n, :, :] = R[n].reciprocal()

                U = ((X*rR[n, :, None]) @ X.mT.conj()) / n_frames
                WU = W @ U

                try:
                    W[:, n, :] = torch.linalg.solve(WU, E[..., n]).conj()
                except torch.linalg.LinAlgError:
                    import warnings
                    warnings.warn("Singular matrix encountered, switching to pseudo-inverse")

                    W[:, n, :] = (WU.pinverse() @ E[..., [n]])[..., 0]

                denom = torch.einsum('im,im->i', torch.einsum('im,imn->in', W[:, n], U), W[:, n].conj())
                W[:, n, :] /= denom[:, None].sqrt()

            demix(W, X, Y)

            cY = Y.mT.conj()
            D = ((X[:, [0]] @ cY) @ (Y @ cY).pinverse()).mT
            W[:] *= D
            Y[:] *= D

            pD = (D*D.conj()).real.permute(1, 0, 2)
            R[:] *= pD
            T[:] *= pD

            Y[:] = self.stft(self.istft(Y.permute(1, 0, 2))).permute(1, 0, 2)
            P[:] = torch.einsum('inj,inj->nij', Y, Y.conj()).real.clamp(min=EPS)

            # cost[epoch+1] = self.calc_cost_fn(P, R, W)

        if proj_back:
            z = projection_back(Y.permute(2, 0, 1), X[:, 0].T)
            Y *= z[..., None].conj()

        ret_val = {'signals': Y.permute(1, 0, 2)}

        if return_filters:
            ret_val['filters'] = W

        if return_scms:
            # (X:imj->1imj,rR:nij->ni1j)->nimj,XmT:ijm=>H:nimm
            H = ((X[None] * rR[:, :, None]) @ X.mT.conj()) / n_frames
            ret_val['scms'] = H

        return ret_val

"""
    - Original paper :
        - D. Kitamura, S. Mogami, Y. Mitsui, N. Takamune, H. Saruwatari, N. Ono, Y. Takahashi,
        and K. Kondo, "Generalized independent low-rank matrix analysis using heavy-tailed distributions
        for blind source separation," EURASIP Journal on Advances in Signal Processing,
        vol. 2018, no. 28, p. 25, May 2018.

    - Original code @ d-kitamura :
        - https://github.com/d-kitamura/ILRMA/blob/master/tILRMA.m
"""
class tILRMA(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def calc_cost_fn(self, P, R, W):
        n_frames = P.size(2)

        log_det_abs = W.det().abs().clamp(min=EPS).log()
        cost = (P/R + R.log()).sum() - 2*n_frames*log_det_abs.sum()

        return cost

    def separate(
        self,
        Xnkl,
        n_src=None,
        n_iter=20,
        n_components=2,
        nu=1,   # degree-of-freedom parameter (1: Cauchy, inf: Gauss)
        p=2,    # signal domain (2: power, 1: amplitude)
        W0=None,
        proj_back=True,
        return_filters=False,
        return_scms=False,
    ):
        n_chan, n_freq, n_frames = Xnkl.shape
        dtype, device = Xnkl.dtype, Xnkl.device

        if n_src is None:
            n_src = n_chan

        assert n_chan == n_src, "There should be as many microphones as sources"

        X = Xnkl.permute(1, 0, 2)

        if W0 is None:
            W = torch.eye(n_src, dtype=dtype, device=device).tile(n_freq, 1, 1)
        else:
            W = W0

        Y = torch.zeros((n_freq, n_src, n_frames), dtype=dtype, device=device)

        def demix(W, X, Y):
            Y[:] = torch.einsum('inm,imj->inj', W, X)

        demix(W, X, Y)

        P = torch.einsum('inj,inj->nij', Y, Y.conj()).real.clamp(min=EPS)
        T = torch.rand((n_src, n_freq, n_components), device=device).clamp(min=EPS)
        V = torch.rand((n_src, n_components, n_frames), device=device).clamp(min=EPS)
        R = T @ V
        rR = R.reciprocal()
        B = (1-(2/(nu+2))) * R**(2/p) + (2/(nu+2)) * P
        E = torch.eye(n_src, dtype=dtype, device=device).tile(n_freq, 1, 1)

        # lambda_aux = torch.zeros(n_src)
        # cost = torch.zeros(n_iter+1)

        # cost[0] = self.calc_cost_fn(P, R, W)

        self.pbar = trange(n_iter)
        for epoch in self.pbar:
            for n in range(n_src):
                T[n, :, :] *= (((P[n]*B[n].reciprocal()*rR[n]) @ V[n].T) / (rR[n] @ V[n].T))**(p/(p+2))
                T[n, :, :] = T[n].clamp(min=EPS)
                R[n, :, :] = T[n] @ V[n]
                rR[n, :, :] = R[n].reciprocal()
                B[n, :, :] = (1-(2/(nu+2))) * R[n]**(2/p) + (2/(nu+2)) * P[n]

                V[n, :, :] *= ((T[n].T @ (P[n]*B[n].reciprocal() * rR[n])) / (T[n].T @ rR[n]))**(p/(p+2))
                V[n, :, :] = V[n].clamp(min=EPS)
                R[n, :, :] = T[n] @ V[n]
                rR[n, :, :] = R[n].reciprocal()
                B[n, :, :] = (1-(2/(nu+2))) * R[n]**(2/p) + (2/(nu+2)) * P[n]

                rzeta = (1 + (2/nu) * (P[n] * rR[n]**(2/p))).reciprocal()
                U = ((1 + 2/nu)/n_frames) * (X * rzeta[:, None] * rR[n, :, None]**(2/p)) @ X.mT.conj()
                WU = W @ U

                try:
                    W[:, n, :] = torch.linalg.solve(WU, E[..., n]).conj()
                except torch.linalg.LinAlgError:
                    import warnings
                    warnings.warn("Singular matrix encountered, switching to pseudo-inverse")

                    W[:, n, :] = (WU.pinverse() @ E[..., [n]])[..., 0]

                denom = torch.einsum('im,im->i', torch.einsum('im,imn->in', W[:, n], U), W[:, n].conj())
                W[:, n, :] /= denom[:, None].sqrt()

            demix(W, X, Y)

            cY = Y.mT.conj()
            D = ((X[:, [0]] @ cY) @ (Y @ cY).pinverse()).mT
            W[:] *= D
            Y[:] *= D

            pD = (D.abs() ** p).permute(1, 0, 2)
            R[:] *= pD
            T[:] *= pD

            # P = torch.einsum('inj,inj->nij', Y, Y.conj()).real.clamp(min=EPS)
            #
            # for n in range(n_src):
            #     lambda_aux[n] = P[n].mean().reciprocal()

            #     W[:, n, :] *= lambda_aux[n].sqrt()
            #     P[n, :, :] *= lambda_aux[n]
            #     R[n, :, :] *= lambda_aux[n]
            #     T[n, :, :] *= lambda_aux[n]

            # cost[epoch+1] = self.calc_cost_fn(P, R, W)

        if proj_back:
            z = projection_back(Y.permute(2, 0, 1), X[:, 0].T)
            Y *= z[..., None].conj()

        ret_val = {'signals': Y.permute(1, 0, 2)}

        if return_filters:
            ret_val['filters'] = W

        if return_scms:
            H = ((X[None] * rR[:, :, None]) @ X.mT.conj()) / n_frames
            ret_val['scms'] = H

        return ret_val
