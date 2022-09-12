from kivy.lang import Builder
from kivy.properties import DictProperty

from batools.app.gui.widgets.container import Container
from batools.utils.audio.wave import load_wave

Builder.load_file(__file__[:-3]+'.kv')


class OffprocessWorkingContainer(Container):
    audio_dict = DictProperty({})

    def on_kv_post(self, *args, **kwargs):
        audio_display = self.ids.audio_display
        audio_toolbar = audio_display.ids.audio_toolbar

        audio_toolbar.root_audio_dict_container = self

    def on_audio_dict(self, instance, value):
        audio_dict = self.audio_dict

        audio_display = self.ids.audio_display
        audio_detail = self.ids.audio_detail

        if audio_dict and audio_dict['data'] is None:
            audio_path = self.audio_dict['path']
            audio_dict['data'], audio_dict['fs'] = load_wave(audio_path)

        audio_display.audio_dict = audio_detail.audio_dict = value

class OffprocessAudioDisplay(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_timeline = self.ids.audio_timeline
        audio_toolbar = self.ids.audio_toolbar

        audio_timeline.audio_dict = audio_toolbar.audio_dict = value

class OffprocessAudioDetail(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        target = self.ids.target
        frog = self.ids.frog

        target.audio_dict = frog.audio_dict = value
