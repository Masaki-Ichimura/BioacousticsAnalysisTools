import torch
import torchaudio
from itertools import combinations

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import *
from kivymd.color_definitions import colors
from kivymd.uix.screen import MDScreen
from kivymd.uix.label.label import MDIcon
from kivymd.uix.selectioncontrol.selectioncontrol import MDCheckbox

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
    checkboxes = []

    def on_audio_dict(self, instance, value):
        if self.audio_dict:
            sep_data, sep_fs = self.audio_dict['data'], self.audio_dict['fs']
            sep_path = self.audio_dict['cache']
            dot_idx = -sep_path[::-1].index('.')-1
            ch_path = sep_path[:dot_idx]+'_ch{:02d}'+sep_path[dot_idx:]

            self.ids.stack_sep.clear_widgets()

            checkboxes = []
            for ch, ch_data in enumerate(sep_data):
                torchaudio.save(
                    filepath=ch_path.format(ch), src=ch_data[None], sample_rate=sep_fs
                )

                audio_miniplot = AudioMiniplot(
                    data=ch_data, fs=sep_fs, path=ch_path.format(ch),
                    size_hint=(1/3, 1/3)
                )

                checkbox_widget = MDCheckbox()
                checkbox_widget.selected_color = checkbox_widget.unselected_color = colors['Blue']['A400']
                checkbox_widget.pos_hint = {'x': .0, 'top': .4}
                checkbox_widget.size_hint = (.25, None)

                audio_miniplot.add_widget(checkbox_widget)
                checkboxes.append(checkbox_widget)

                self.ids.stack_sep.add_widget(audio_miniplot)

            self.checkboxes = checkboxes

    def select(self):
        indices = [checkbox.active for checkbox in self.checkboxes]
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
            ana_datas, ana_fs = self.audio_dict['data'], self.audio_dict['fs']

            if ana_datas.size(0) > 1:
                combs = combinations(range(ana_datas.size(0)), 2)

                sep_miniplots = self.root_tab.ids.select.ids.stack_sep.children
                sep_datas = [mp.audio_data for mp in sep_miniplots]

                sct_indices = [
                    [torch.equal(ana_data, sep_data) for sep_data in sep_datas].index(True)
                    for ana_data in ana_datas
                ]
                sct_miniplots = [sep_miniplots[idx] for idx in sct_indices]
                sct_miniplots = [
                    AudioMiniplot(
                        data=mp.audio_data, fs=mp.audio_fs, path=mp.audio_path, figure=mp.figure,
                        size=mp.size, size_hint=(None, None)
                    )
                    for mp in sct_miniplots
                ]

                self.ids.box_ana.clear_widgets()
                for i, mp in enumerate(sct_miniplots):
                    i_widget = MDIcon(
                        icon=f'numeric-{i}-box',
                        theme_text_color='Custom', text_color=colors['Blue']['A400']
                    )
                    i_widget.pos_hint = {'x': .05, 'y': .1}
                    i_widget.size_hint = (.25, None)
                    mp.add_widget(i_widget)
                    self.ids.box_ana.add_widget(mp)

                peaks_tmp = {}
                for comb in combs:
                    A_idx, B_idx = comb
                    At, Bt = ana_datas[A_idx], ana_datas[B_idx]

                    res = check_synchronization(At, Bt, ana_fs)
                    peaks_dict = res.pop('peaks')

                    if A_idx not in peaks_tmp:
                        peaks_tmp[A_idx] = peaks_dict['A']
                    if B_idx not in peaks_tmp:
                        peaks_tmp[B_idx] = peaks_dict['B']

                    print(comb, res)

                for idx, peaks in peaks_tmp.items():
                    audio_miniplot = self.ids.box_ana.children[idx]
                    audio_data = audio_miniplot.audio_data.squeeze()
                    ax_wave = audio_miniplot.figure.axes[0]
                    ax_wave.plot(peaks, audio_data[peaks], marker='x', color='g', linewidth=0)


                    # fig, ax = plt.subplots()
                    # ax.hist(phis, bins=8, range=(0, 2*torch.pi))
                    # ax.set_xticks(torch.arange(0, 2*torch.pi+1e-6, torch.pi/2))
                    # ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
