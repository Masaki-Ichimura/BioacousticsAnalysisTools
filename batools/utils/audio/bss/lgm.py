"""
    LGMの実装は省略
"""
import torch
from torchnmf.nmf import NMF
from tqdm import trange

from .base import tf_bss_model_base
from .fastmnmf import FastMNMF


class LGM(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def forward(self, xnt, **separate_args):
        Xnkl = self.stft(xnt)
        ret_val = self.separate(Xnkl, **separate_args)
        Ymnkl, losses = ret_val if type(ret_val) == tuple else (ret_val, None)
        ymnt = self.istft(Ymnkl, xnt.shape[-1])

        return ymnt if losses is None else (ymnt, losses)

    def separate(self, Xnkl):
        raise NotImplementedError


"""
    S. Arberet et al., "Nonnegative matrix factorization and spatial covariance model for under-determined reverberant audio source separation,"
    10th International Conference on Information Science,
    ISSPA, 2010, pp. 1-4
"""
class NMFLGM(LGM):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def separate(
        self,
        Xnkl,
        n_src=None, n_iter=10, n_components=4, sigma2=1e-2, eps=1e-10,
        return_losses=False
    ):

        n_chan, n_freq, n_frames = Xnkl.shape
        dtype, device = Xnkl.dtype, Xnkl.device

        if n_src is None:
            n_src = n_chan

        X_ftc = Xnkl.permute(1, 2, 0)
        I = torch.eye(n_chan, device=device)
        I_eps = eps*I

        def normalize_RV(P):
            if P.is_complex():
                Pnorm = (P*P.conj()).real.sum((-2, -1), keepdim=True)
            else:
                Pnorm = (P**2).sum((-2, -1), keepdim=True)

            return P / Pnorm.clip(min=eps)

        def normalize_W(W):
            Wnorm = W.sum(1, keepdim=True)
            return W / Wnorm.clip(min=eps)

        def normalize_H(H):
            Hnorm = H.sum(2, keepdim=True)
            return H / Hnorm.clip(min=eps)

        def trace(A):
            tr = lambda x: torch.diagonal(x, offset=0, dim1=-1, dim2=-2).sum(-1)
            trA = tr(A)
            return trA

        def determinant(A):
            det = lambda x: torch.det(x)
            detA = det(A)
            return detA

        # init parameters using MNMF
        mnmf = FastMNMF(**self.stft_args)
        V_nft = mnmf.separate(
            Xnkl.cpu(), n_src=n_src, n_iter=30, n_components=2
        ).abs()
        V_nft /= V_nft.max()

        Rx_ftcc = torch.einsum('ftc,ftd->ftcd', X_ftc, X_ftc.conj())
        R_nfcc = torch.einsum('nft,ftcd->nftcd', V_nft, Rx_ftcc).mean(2)

        nmf = NMF(V_nft.shape[1:], n_components)
        W_nfk, H_nkt = [], []
        for V_ft in V_nft.cpu():
            nmf.fit(V_ft)
            W_nfk.append(nmf.H.detach())
            H_nkt.append(nmf.W.detach().mT)

        W_nfk = normalize_W(torch.stack(W_nfk).to(device))
        H_nkt = normalize_H(torch.stack(H_nkt).to(device))
        V_nft = normalize_RV(torch.bmm(W_nfk, H_nkt))

        Rb_fcc = sigma2 * \
            torch.eye(n_chan, dtype=dtype, device=device).tile(n_freq, 1, 1)
        R_nfcc = normalize_RV(R_nfcc)

        self.pbar = trange(n_iter)

        loss_list = []
        for epoch in self.pbar:
            # E-step
            V_nft = V_nft if epoch == 0 else torch.bmm(W_nfk, H_nkt)
            V_nft = V_nft.clip(min=0)

            R_nftcc = torch.einsum('nft,nfcd->nftcd', V_nft, R_nfcc)
            Rx_ftcc = R_nftcc.sum(0) + Rb_fcc[:, None]
            Rx_inv = torch.linalg.pinv(Rx_ftcc + I_eps)
            R_nkftcc = torch.einsum('nfk,nkt,nfcd->nkftcd', W_nfk, H_nkt, R_nfcc)

            G = R_nftcc @ Rx_inv
            Qh = torch.einsum('nftcd,ftd->nftc', G, X_ftc)
            Rh_nftcc = \
                torch.einsum('nftc,nftd->nftcd', Qh, Qh.conj()) \
                + (I - G) @ R_nftcc

            G = R_nkftcc @ Rx_inv
            Qh = torch.einsum('nkftcd,ftd->nkftc', G, X_ftc)
            Rh_nkftcc = \
                torch.einsum('nkftc,nkftd->nkftcd', Qh, Qh.conj()) \
                + (I - G) @ R_nkftcc

            G = Rb_fcc[:, None] @ Rx_inv
            Qh = torch.einsum('ftcd,ftd->ftc', G, X_ftc)
            Rbh_ftcc = \
                torch.einsum('ftc,ftd->ftcd', Qh, Qh.conj()) \
                + (I - G) @ Rb_fcc[:, None]

            # M-step
            R_nfcc = (Rh_nftcc / V_nft.clip(min=eps)[..., None, None]).mean(2)
            R_inv = torch.linalg.pinv(R_nfcc + I_eps)

            Vh_nkft = trace(
                torch.einsum('nfcd,nkftde->nkftce', R_inv, Rh_nkftcc)
            ).real.clip(min=0) / n_chan
            W_nfk, H_nkt = (
                normalize_W((Vh_nkft/H_nkt.clip(min=eps)[..., None, :]).mean(3).mT),
                normalize_H((Vh_nkft/W_nfk.clip(min=eps).mT[..., None]).mean(2))
            )

            Rb_fcc = Rbh_ftcc.mean(1)*I

            # loss
            if return_losses:
                loss = (
                    torch.einsum(
                        'ftc,ftcd,ftd->ft', X_ftc.conj(), Rx_inv, X_ftc
                    ).real.abs() \
                    + torch.log(determinant(Rx_ftcc).real.abs() + eps)
                ).sum()
                loss_list.append(loss.item())

        V_nft = torch.bmm(W_nfk, H_nkt).clip(min=0)
        R_nftcc = torch.einsum('nft,nfcd->nftcd', V_nft, R_nfcc)
        Rx_ftcc = R_nftcc.sum(0) + Rb_fcc[:, None]
        Rx_inv = torch.linalg.pinv(Rx_ftcc + I_eps)
        Y = torch.einsum('nftcd,ftd->nftc', R_nftcc@Rx_inv, X_ftc)

        Ymnkl = Y.permute(0, 3, 1, 2)
        Ymnkl = Ymnkl * (
            Xnkl.abs().max() / Ymnkl.reshape(n_src, -1).abs().max(-1).values
        )[:, None, None, None]

        if return_losses:
            return Ymnkl, loss_list
        else:
            return Ymnkl
