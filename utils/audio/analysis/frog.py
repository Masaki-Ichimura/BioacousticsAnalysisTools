import torch
from scipy.signal import find_peaks


def check_synchronization(
    At: torch.Tensor, Bt: torch.Tensor, sample_rate: int,
    call_interval_ms: int=150, minimum_amplitude_rate: float=.5
):
    At_abs, Bt_abs = At.abs(), Bt.abs()

    A_peaks, _ = find_peaks(
        At_abs,
        distance=sample_rate*call_interval_ms//1000,
        height=(At_abs.max()*minimum_amplitude_rate).item()
    )
    A_peaks = torch.from_numpy(A_peaks)
    B_peaks, _ = find_peaks(
        Bt_abs,
        distance=sample_rate*call_interval_ms//1000,
        height=(Bt_abs.max()*minimum_amplitude_rate).item()
    )
    B_peaks = torch.from_numpy(B_peaks)

    phis = []
    for i, A_t2 in enumerate(A_peaks[1:]):
        A_t1 = A_peaks[i]
        B_tx = B_peaks[torch.logical_and(A_t1<B_peaks, B_peaks<A_t2)]

        if B_tx.numel() == 1:
            B_t1 = B_tx
            phi = 2 * torch.pi * (B_t1-A_t1) / (A_t2-A_t1)
            phis.append(phi)

    phis = torch.tensor(phis)
    n = phis.numel()

    if n != 0:
        theta0 = torch.tensor(torch.pi)
        mean_x, mean_y = phis.cos().mean(), phis.sin().mean()

        if mean_x > 0:
            mean_p = (mean_y/mean_x).arctan()
        else:
            mean_p = (mean_y/mean_x).arctan() + torch.pi

        theta = (mean_x.square()+mean_y.square()).sqrt() * (mean_p-theta0).cos()
        V = (2 * n)**.5 * theta.item()
    else:
        V = None

    return dict(n=n, V=V, phis=phis, peaks={'A': A_peaks, 'B': B_peaks})
