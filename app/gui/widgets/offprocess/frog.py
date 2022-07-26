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

    def on_audio_dict(self, instance, value):
        separate = self.ids.separate

        separate.audio_dict = value


class FrogSeparate(MDScreen):
    audio_dict = DictProperty({})
    mode = 'ilrma'

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



class FrogSelect(MDScreen):
    pass


class FrogAnalysis(MDScreen):
    pass
