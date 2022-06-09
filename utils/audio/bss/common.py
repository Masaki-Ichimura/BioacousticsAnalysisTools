"""
    This module is based on PyAudioAnalysis (MIT Licence)
        - https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.bss.html
"""
import torch


def projection_back(Y, ref, clip_up=None, clip_down=None):

    num = (ref[:, :, None].conj() * Y).sum(dim=0)
    denom = (Y*Y.conj()).real.sum(dim=0)

    c = torch.ones(num.shape, dtype=Y.dtype, device=Y.device)
    I = denom > 0.0
    c[I] = num[I] / denom[I]

    if clip_up is not None:
        I = torch.logical_and(c.abs() > clip_up, c.abs() > 0)
        c[I] *= clip_up / c[I].abs()

    if clip_down is not None:
        I = torch.logical_and(c.abs() < clip_down, c.abs() > 0)
        c[I] *= clip_down / c[I].abs()
    return c


def sparir(
    G,
    S,
    weights=torch.tensor([]),
    gini=0,
    maxiter=50,
    tol=10,
    alpha=10,
    alphamax=1e5,
    alphamin=1e-7,
):

    n_freq = G.shape[0]

    y = torch.concatenate((G[S].real, G[S].imag), dim=0)
    M = y.shape[0]

    if gini == 0:  # if no initialization is given
        g = torch.zeros((n_freq, 1))
        g[0] = 1
    else:
        g = gini

    if weights.size == 0:
        tau = torch.sqrt(n_freq) / (y.conj().T.dot(y))
        tau = tau * torch.exp(
            0.11 * (torch.arange(1.0, n_freq + 1.0).T).abs() ** 0.3
        )
        tau = tau.T
    elif weights.shape[0] == 1:
        tau = torch.ones((n_freq, 1)) * weights
    else:
        tau = torch.tile(weights.T, (1, 1)).reshape(n_freq)

    def soft(x, T):
        if torch.sum(T.abs().flatten()) == 0:
            u = x
        else:
            u = torch.max(x.abs() - T, 0)
            u = u / (u + T) * x
        return u

    aux = torch.zeros((n_freq, 1), dtype=torch.complex)
    G = torch.fft.fft(g.flatten())
    Ag = torch.concatenate((G[S].real, G[S].imag), dim=0)
    r = Ag - y.flatten()  # instead of r = A * g - y
    aux[S] = (r[0 : M // 2] + 1j * r[M // 2 :])[:, None]
    gradq = n_freq * torch.fft.irfft(aux.flatten(), n_freq)  # instead of gradq = A'*r
    gradq = gradq[:, None]
    support = g != 0
    iter_ = 0  # initial iteration value

    # Define stopping criteria
    crit = torch.zeros((maxiter, 1))
    criterion = -tau[support] * torch.sign(g[support]) - gradq[support]
    crit[iter_] = torch.sum(criterion ** 2)

    while (crit[iter_] > tol) and (iter_ < maxiter - 1):
        # Update gradient
        prev_r = r
        prev_g = g
        g = soft(prev_g - gradq * (1.0 / alpha), tau / alpha)
        dg = g - prev_g
        DG = torch.fft.fft(dg.flatten())
        Adg = torch.concatenate((DG[S].real, DG[S].imag), dim=0)
        r = prev_r + Adg.flatten()  # faster than A * g - y
        dd = torch.dot(dg.flatten().conj().T, dg.flatten())
        dGd = torch.dot(Adg.flatten().conj().T, Adg.flatten())
        alpha = min(alphamax, max(alphamin, dGd / (torch.finfo(torch.float32).eps + dd)))
        iter_ += 1
        support = g != 0
        aux[S] = (r[0 : M // 2] + 1j * r[M // 2 :])[:, None]
        gradq = n_freq * torch.fft.irfft(aux.flatten(), n_freq)
        gradq = gradq[:, None]
        # Update stopping criteria
        criterion = -tau[support] * torch.sign(g[support]) - gradq[support]
        crit[iter_] = sum(criterion ** 2) + sum(
            abs(gradq[~support]) - tau[~support] > tol
        )

    return g.flatten()
