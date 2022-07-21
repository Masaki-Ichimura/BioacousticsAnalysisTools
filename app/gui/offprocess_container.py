from kivy.lang import Builder
from kivy.properties import *

from app.gui.widgets.container import Container

Builder.load_file(__file__[:-3]+'.kv')


class OffprocessContainer(Container):
    pass

class OffprocessWorkingContainer(Container):
    audio_dict = DictProperty({})

    def on_kv_post(self, *arg, **kwargs):
        audio_display = self.ids.target.ids.audio_display
        audio_toolbar = audio_display.ids.audio_toolbar

        audio_toolbar.root_audio_dict_container = self

    def on_audio_dict(self, instance, value):
        target = self.ids.target

        target.audio_dict = value
