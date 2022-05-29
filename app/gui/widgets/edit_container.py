from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from kivy.uix.widget import Widget

from app.gui.main_container import MainContainer
from utils.audio import load_wav
from utils.plot import show_spec, show_wav

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_container.kv')


class EditContainer(MainContainer):
    pass

class EditWorkingContainer(MainContainer):
    audio_file = StringProperty('')
    audio_data = None
    audio_fs = NumericProperty(0)

    def on_audio_file(self, instance, value):
        audio_toolbar = self.ids.audio_display.ids.audio_toolbar
        audio_timeline = self.ids.audio_display.ids.audio_timeline

        audio_data, audio_fs = load_wav(self.audio_file)

        self.audio_data, self.audio_fs = audio_data, audio_fs

        audio_timeline.audio_file = value

class EditAudioDisplay(MainContainer):
    pass

class EditAudioDetail(MainContainer):
    pass
