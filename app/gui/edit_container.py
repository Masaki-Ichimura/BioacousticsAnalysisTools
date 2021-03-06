from kivy.app import App
from kivy.lang import Builder
from kivy.properties import *
from kivy.uix.widget import Widget

from app.gui.widgets.container import Container
from app.kivy_utils import TorchTensorProperty
from utils.audio.wave import load_wave

Builder.load_file(__file__[:-3]+'.kv')


class EditContainer(Container):
    def on_kv_post(self, *args, **kwargs):
        app = App.get_running_app()
        self.app = app

class EditWorkingContainer(Container):
    audio_dict = DictProperty({})

    def on_kv_post(self, *arg, **kwargs):
        self.edit_container = self.parent.parent

        audio_display = self.ids.audio_display
        audio_toolbar = audio_display.ids.audio_toolbar

        audio_toolbar.root_audio_dict_container = self

    def on_audio_dict(self, instance, value):
        audio_dict = self.audio_dict

        audio_display = self.ids.audio_display
        audio_detail = self.ids.audio_detail

        if audio_dict and audio_dict['data'] is None:
            audio_dict['data'], audio_dict['fs'] = load_wave(self.audio_dict['path'])

        audio_display.audio_dict = audio_detail.audio_dict = value


class EditAudioDisplay(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_timeline = self.ids.audio_timeline
        audio_toolbar = self.ids.audio_toolbar

        audio_timeline.audio_dict = audio_toolbar.audio_dict = value


class EditAudioDetail(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        silence_removal = self.ids.silence_removal

        silence_removal.audio_dict = value
