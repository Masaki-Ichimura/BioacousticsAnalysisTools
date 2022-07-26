from kivy.app import App
from kivy.lang import Builder
from kivy.properties import *
from kivymd.uix.screen import MDScreen

from app.gui.widgets.container import Container
from app.gui.widgets.tab import Tab
from utils.audio.bss.auxiva import AuxIVA
from utils.audio.bss.fastmnmf import FastMNMF
from utils.audio.bss.ilrma import ILRMA

Builder.load_file(__file__[:-3]+'.kv')


class FrogTab(Tab):
    audio_dict = DictProperty({})

    def on_kv_post(self, *args, **kwargs):
        self.ids.nav_drawer.set_state('open')

        tabs = [self.ids.separate, self.ids.select, self.ids.analysis]

        for tab in tabs:
            tab.root_tab = self

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

            org_tag = self.audio_dict['tag']
            org_data, org_fs = self.audio_dict['data'], self.audio_dict['fs']

            sep_fn = self.get_func()

            sep_data = sep_fn(org_data, n_iter=30)

            sep_tag = f'{org_tag}_{self.mode}'
            sep_cache = f'{cache_dir.name}/sep_{org_tag}.wav'
            sep_dict = dict(
                tag=sep_tag, path=None, cache=sep_cache,
                data=sep_data, fs=org_fs, ch=-1
            )

            self.root_tab.ids.select.audio_dict = sep_dict

class FrogSelect(MDScreen):
    audio_dict = DictProperty({})
    root_tab = None

class FrogAnalysis(MDScreen):
    root_tab = None
