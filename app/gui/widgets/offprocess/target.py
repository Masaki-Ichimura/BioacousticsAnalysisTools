from kivy.lang import Builder
from kivy.properties import *

from app.gui.widgets.container import Container
from app.gui.widgets.tab import Tab

Builder.load_file(__file__[:-3]+'.kv')


class TargetTab(Tab):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_display = self.ids.audio_display

        audio_display.audio_dict = value


class TargetAudioDisplay(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_timeline = self.ids.audio_timeline
        audio_toolbar = self.ids.audio_toolbar

        audio_timeline.audio_dict = audio_toolbar.audio_dict = value
