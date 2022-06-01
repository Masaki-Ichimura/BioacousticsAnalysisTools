from kivy.lang import Builder
from kivy.properties import *
from kivy.uix.widget import Widget
from kivymd.uix.tab import MDTabsBase
from kivymd.uix.floatlayout import MDFloatLayout

from app.gui.widgets.container import Container
from utils.audio.wave import load_wave

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_container.kv')


class EditContainer(Container):
    pass

class EditWorkingContainer(Container):
    audio_file = StringProperty('')
    audio_data = None
    audio_fs = NumericProperty(0)

    def on_audio_file(self, instance, value):
        if not value:
            audio_data = None
            return None

        audio_toolbar = self.ids.audio_display.ids.audio_toolbar
        audio_timeline = self.ids.audio_display.ids.audio_timeline

        audio_data, audio_fs = load_wave(self.audio_file)

        self.audio_data, self.audio_fs = audio_data, audio_fs

        audio_timeline.audio_file = value

class EditAudioDisplay(Container):
    pass

class EditAudioDetail(Container):
    pass

class Tab(MDFloatLayout, MDTabsBase):
    pass

'''
    以下仮置き，コード移動不可避
'''
import torch
from utils.audio.silence import *

stft_dict = dict(
    n_fft=1024,
    hop_length=256,
    window=torch.hann_window(1024+2)[1:-1]
)

stft = lambda x: torch.stft(x, return_complex=True, **stft_dict)
istft = lambda x, l: torch.istft(x, length=l, **stft_dict)

class RemovingSilenceTab(Tab):
    def ok_button_clicked(self):
        freq_dict = {
            'freq_high': int(self.ids.freq_high.text),
            'freq_low': int(self.ids.freq_low.text)
        }
        args_dict = {
            'min_silence_len': int(self.ids.min_silence_len.text),
            'silence_thresh': float(self.ids.silence_thresh.text),
            'seek_step': int(self.ids.seek_step.text),
        }
        print(freq_dict)
        print(args_dict)

        audio_detail = self.parent.parent.parent.parent.parent.parent
        working_container = audio_detail.parent.parent
        audio_timeline = working_container.ids.audio_display.ids.audio_timeline

        audio_data = working_container.audio_data
        audio_fs = working_container.audio_fs

        if not working_container.audio_file:
            return None

        xnt = audio_data
        xnkf = stft(xnt)
        k = torch.fft.rfftfreq(stft_dict['n_fft'])*audio_fs
        hk = torch.logical_or(k<freq_dict['freq_low'], k>freq_dict['freq_high'])
        xnkf[:, hk] = 0
        ynt = istft(xnkf, audio_data.shape[-1])
        silence_sections = detect_silence(ynt, audio_fs, **args_dict)

        print(silence_sections)


        d_min, d_max = xnt.mean(0).min().item(), xnt.mean(0).max().item()
        d_min, d_max = min(d_min, -abs(d_max)), max(d_max, abs(d_min))
        silence = d_max*torch.ones(int(audio_data.shape[-1]/audio_fs*1000))
        for sec in silence_sections:
            silence[sec[0]:sec[1]] = d_min

        ax_wave = audio_timeline.fig_wave.axes[0]

        if len(ax_wave.lines)>1:
            ax_wave.lines[-1].remove()

        ax_wave.plot(
            torch.linspace(0, audio_data.shape[-1]/audio_fs, steps=len(silence)),
            silence,
            color='r'
        )

        audio_timeline.fig_wave.canvas.draw()
