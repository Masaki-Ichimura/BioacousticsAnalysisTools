import torch

from kivy.lang import Builder
from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivymd.uix.menu import MDDropdownMenu

from app.gui.widgets.container import Container

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/config_container.kv')


class ConfigContainer(Container):
    def on_kv_post(self, *arg, **kwargs):
        self.ids.nav_drawer.set_state('open')

        self.stft_window_menu = MDDropdownMenu(
            caller=self.ids.stft_window,
            hor_growth='right',
            width_mult=4,
            items=[
                dict(
                    viewclass='OneLineListItem',
                    text=k,
                    height=dp(54),
                    on_release=lambda x=k: self.set_stft_window(x)
                )
                for k in ['bartlett', 'blackman', 'hamming', 'hann', 'kaiser']
            ]
        )

    def set_stft_window(self, window):
        self.ids.stft_window.set_text(self.ids.stft_window, window)
        self.stft_window_menu.dismiss()

    def get_stft_args(self):
        n_fft = int(self.ids.stft_n_fft.text)

        hop_length = int(self.ids.stft_hop_length.text)

        window = self.ids.stft_window.text
        if window == 'bartlett':
            window = torch.bartlett_window(n_fft)
        elif window == 'blackman':
            window = torch.blackman_window(n_fft)
        elif window == 'hamming':
            window = torch.hamming_window(n_fft)
        elif window == 'hann':
            window = torch.hann_window(n_fft)
        elif window == 'kaiser':
            window = torch.kaiser_window(n_fft)
        else:
            window = None

        args = dict(
            n_fft=n_fft,
            hop_length=hop_length,
            window=window
        )
        return args
