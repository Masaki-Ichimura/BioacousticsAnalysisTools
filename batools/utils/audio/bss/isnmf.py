import torch
from tqdm import trange

from .base import tf_bss_model_base

class ISNMF(tf_bss_model_base):
    def __init__(self, **stft_args):
        super().__init__(**stft_args)

    def forward(self, xt, **separate_args):
        Xkl = self.stft(xt)
        Ynkl = self.separate(Xkl, **separate_args)
        ynt = self.istft(Ynkl, xt.shape[-1])

        return ynt

    def separate(self, Xkl, n_src=None, n_iter=20):
        NotImplementedError
