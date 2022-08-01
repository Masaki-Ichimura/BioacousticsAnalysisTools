import torch
import torchaudio
from itertools import combinations

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import *
from kivymd.uix.screen import MDScreen

from app.gui.widgets.container import Container
from app.gui.widgets.tab import Tab
from app.gui.widgets.audiodisplay import AudioMiniplot
from app.kivy_utils import TorchTensorProperty
from utils.audio.bss.auxiva import AuxIVA
from utils.audio.bss.fastmnmf import FastMNMF
from utils.audio.bss.ilrma import ILRMA
from utils.audio.analysis.frog import check_synchronization

Builder.load_file(__file__[:-3]+'.kv')

from kivy.uix.button import Button

class FrogTab(Tab):
    audio_dict = DictProperty({})

    def on_kv_post(self, *args, **kwargs):
        self.ids.nav_drawer.set_state('open')

    def on_audio_dict(self, instance, value):
        separate = self.ids.separate
        separate.audio_dict = value

class FrogSeparate(MDScreen):
    audio_dict = DictProperty({})
    mode = 'ilrma'

    root_tab = None

    def get_func(self):
        app = App.get_running_app()
        config_container = app.links['config_container']
        stft_args = config_container.get_stft_args()

        if self.mode == 'auxiva':
            func = AuxIVA(**stft_args)
        elif self.mode == 'fastmnmf':
            func = FastMNMF(**stft_args)
        elif self.mode == 'ilrma':
            func = ILRMA(**stft_args)

        return func

    def separate(self):
        if self.audio_dict:
            app = App.get_running_app()
            cache_dir = app.tmp_dir

            sep_fn = self.get_func()

            sep_data, sep_fs = sep_fn(self.audio_dict['data'], n_iter=30), self.audio_dict['fs']
            sep_tag = f'separate_{self.mode}_{self.audio_dict["tag"]}'
            sep_cache = f'{cache_dir.name}/{sep_tag}.wav'
            sep_dict = dict(
                tag=sep_tag, path=None, cache=sep_cache, data=sep_data, fs=sep_fs, ch=-1
            )

            self.root_tab.ids.select.audio_dict = sep_dict

class FrogSelect(MDScreen):
    audio_dict = DictProperty({})
    root_tab = None

    def on_audio_dict(self, instance, value):
        if self.audio_dict:
            sep_data, sep_fs = self.audio_dict['data'], self.audio_dict['fs']
            sep_path = self.audio_dict['cache']
            dot_idx = -sep_path[::-1].index('.')-1
            ch_path = sep_path[:dot_idx]+'_ch{:02d}'+sep_path[dot_idx:]

            self.ids.grid_seps.clear_widgets()

            for ch, ch_data in enumerate(sep_data):
                torchaudio.save(
                    filepath=ch_path.format(ch), src=ch_data[None], sample_rate=sep_fs
                )
                audio_miniplot = AudioMiniplot(
                    data=ch_data, fs=sep_fs, path=ch_path.format(ch),
                    size=(self.width/3, self.height/3)
                )
                self.ids.grid_seps.add_widget(audio_miniplot)

    def select(self):
        indices = [child.checkbox.active for child in self.ids.grid_seps.children]
        if self.audio_dict and any(indices):
            app = App.get_running_app()
            cache_dir = app.tmp_dir

            sct_tag = f'select_{self.audio_dict["tag"]}'
            sct_data, sct_fs = self.audio_dict['data'][indices], self.audio_dict['fs']
            sct_cache = f'{cache_dir.name}/{sct_tag}.wav'

            sct_dict = dict(
                tag=sct_tag, path=None, cache=sct_cache, data=sct_data, fs=sct_fs, ch=-1
            )

            self.root_tab.ids.analysis.audio_dict = sct_dict

class FrogAnalysis(MDScreen):
    audio_dict = DictProperty({})
    root_tab = None

    def on_audio_dict(self, instance, value):
        if self.audio_dict:
            ana_data, ana_fs = self.audio_dict['data'], self.audio_dict['fs']

            if ana_data.size(0) > 1:
                combs = combinations(range(ana_data.size(0)), 2)

                for comb in combs:
                    A_idx, B_idx = comb
                    At, Bt = ana_data[A_idx], ana_data[B_idx]

                    res = check_synchronization(At, Bt, ana_fs)
                    peaks_dict = res.pop('peaks')
                    A_peaks, B_peaks = peaks_dict['A'], peaks_dict['B']

                    print(comb, res)

                    # fig, ax = plt.subplots()
                    # ax.hist(phis, bins=8, range=(0, 2*torch.pi))
                    # ax.set_xticks(torch.arange(0, 2*torch.pi+1e-6, torch.pi/2))
                    # ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
