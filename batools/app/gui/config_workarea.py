import torch

from kivy.lang import Builder
from kivy.metrics import dp
from kivymd.uix.menu import MDDropdownMenu

from batools.app.gui.widgets.container import Container

Builder.load_file(__file__[:-3]+'.kv')


class ConfigWorkingContainer(Container):
    def on_kv_post(self, *arg, **kwargs):
        self.ids.nav_drawer.set_state('open')

        def set_stft_window(window):
            self.ids.stft_window.set_text(self.ids.stft_window, window)
            self.stft_window_menu.dismiss()

        stft_windows = ['bartlett', 'blackman', 'hamming', 'hann', 'kaiser']
        self.stft_window_menu = MDDropdownMenu(
            caller=self.ids.stft_window,
            hor_growth='right',
            width_mult=4,
            items=[
                dict(
                    viewclass='OneLineListItem',
                    text=window,
                    height=dp(54),
                    on_release=lambda x=window: set_stft_window(x)
                )
                for window in stft_windows
            ]
        )

        # def set_ssl_mic_array_type(mic_array_type):
        #     self.ids.ssl_circ_direction.disabled = False if mic_array_type == '円形' else True

        #     self.ids.ssl_mic_array_type.set_text(self.ids.ssl_mic_array_type, mic_array_type)
        #     self.ssl_mic_array_type_menu.dismiss()

        # ssl_mic_array_types = ['円形', '直線(未実装)']
        # self.ssl_mic_array_type_menu = MDDropdownMenu(
        #     caller=self.ids.ssl_mic_array_type,
        #     hor_growth='right',
        #     width_mult=4,
        #     items=[
        #         dict(
        #             viewclass='OneLineListItem',
        #             text=mic_array_type,
        #             height=dp(54),
        #             on_release=lambda x=mic_array_type: set_ssl_mic_array_type(x)
        #         )
        #         for mic_array_type in ssl_mic_array_types
        #     ]
        # )

        def set_ssl_circ_direction(circ_direction):
            self.ids.ssl_circ_direction.set_text(self.ids.ssl_circ_direction, circ_direction)
            self.ssl_circ_direction_menu.dismiss()

        ssl_circ_directions = ['左(反時計回り)', '右(時計回り)']
        self.ssl_circ_direction_menu = MDDropdownMenu(
            caller=self.ids.ssl_circ_direction,
            hor_growth='right',
            width_mult=4,
            items=[
                dict(
                    viewclass='OneLineListItem',
                    text=circ_direction,
                    height=dp(54),
                    on_release=lambda x=circ_direction: set_ssl_circ_direction(x)
                )
                for circ_direction in ssl_circ_directions
            ]
        )

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

        args = dict(n_fft=n_fft, hop_length=hop_length, window=window)

        return args

    def get_ssl_args(self):
        args = dict(n_fft=int(self.ids.stft_n_fft.text))

        if self.ids.ssl_circ_checkbox.active:
            args['mic_array'] = 'circ'
            if self.ids.ssl_circ_direction.text[0] == '左':
                args['circ_direction'] = 'left'
            else:
                args['circ_direction'] = 'right'
            args['circ_radius'] = int(self.ids.ssl_circ_radius.text)/100

        return args

    def get_save_args(self):
        if self.ids.save_normalization.state == 'down':
            normalization = True
        else:
            normalization = None

        args = dict(normalization=normalization)

        return args

    def get_notify(self, subject):
        if subject == 'separate':
            checkbox = self.ids.separate_notify

        if checkbox.state == 'down':
            return True
        else:
            return False