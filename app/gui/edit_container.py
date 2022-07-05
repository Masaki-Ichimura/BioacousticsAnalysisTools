from kivy.lang import Builder
from kivy.properties import *
from kivy.uix.widget import Widget

from app.gui.widgets.container import Container
from utils.audio.wave import load_wave

Builder.load_file(__file__[:-3]+'.kv')


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

        audio_detail = self.ids.audio_detail
        silence_removal = audio_detail.ids.silence_removal

        file_dir = self.audio_file
        file_name = file_dir.split('/')[-1]
        audio_data, audio_fs = load_wave(file_dir)

        self.audio_data, self.audio_fs = audio_data, audio_fs

        audio_timeline.audio_file = value
        silence_removal.ids.tag.text = file_name[:-file_name[::-1].index('.')-1]


class EditAudioDisplay(Container):
    pass

class EditAudioDetail(Container):
    pass
