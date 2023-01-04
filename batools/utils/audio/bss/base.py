import torch
import torchaudio
from nara_wpe import torch_wpe
from collections import ChainMap
from tqdm import trange

class tf_bss_model_base(torch.nn.Module):
    """
    Base class of BSS model in time-frequency domain.
    """
    def __init__(self, **args):
        """
        Parameters
        ----------
        n_fft : int, default 400
        win_length : int or None, default n_fft
        hop_length : int or None, default n_fft//2
        pad : int, default 0
        window_fn : Callable[.., Tensor], default torch.hann_window
        normalized : bool, default False
        wkwargs : dict or None, default None
        center : bool, default True
        pad_mode : str, default 'reflect'
        onesided : bool, default True

        See Also
        --------
        torchaudio.transforms.Spectrogram : Create a spectrogram from a audio signal.
        torchaudio.transforms.InverseSpectrogram : Create an inverse spectrogram to recover an audio signal from a spectrogram.

        Notes
        -----
        These parameters are used for STFT and iSTFT separating signal.
        """
        super(tf_bss_model_base, self).__init__()

        spec_fn = torchaudio.transforms.Spectrogram
        spec_args = {
            k: v for k, v in args.items()
            if k in spec_fn.__init__.__annotations__.keys()
        }
        spec_args['power'] = None
        self.stft = spec_fn(**spec_args)

        ispec_fn = torchaudio.transforms.InverseSpectrogram
        ispec_args = {
            k: v for k, v in args.items()
            if k in ispec_fn.__init__.__annotations__.keys()
        }
        self.istft = ispec_fn(**ispec_args)

        self.stft_args = dict(ChainMap(spec_args, ispec_args))

        self.pbar = trange(0)

    def forward(self, xnt, **separate_args):
        Xnkl = self.stft(xnt)

        wpe = separate_args.pop('wpe') if 'wpe' in separate_args else False

        if wpe:
            print('dereverbed')
            Xnkl = torch_wpe.wpe_v6(Xnkl.permute(1, 0, 2)).permute(1, 0, 2)

        ret_val = self.separate(Xnkl, **separate_args)

        Ynkl = ret_val['signals']
        ynt = self.istft(Ynkl, xnt.shape[-1])

        if len(ret_val) == 1:
            return ynt
        else:
            ret_val['signals'] = ynt
            return ret_val

    def separate(self, Xnkl):
        raise NotImplementedError
