from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import StringProperty, ObjectProperty, NumericProperty
from kivy.core.audio import SoundLoader

from app.gui.main_container import MainContainerWidget

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/audiotimeline.kv')


class AudioTimelineWidget(MainContainerWidget):
    pass

class AudioToolbarWidget(MainContainerWidget):
    audio_file = StringProperty('')
    sound = ObjectProperty(None)
    audio_pos = NumericProperty(0)

    def set_audio(self):
        self.sound = SoundLoader.load(self.audio_file)

    def play(self):
        if self.sound.state == 'stop':
            self.sound.seek(self.audio_pos)
            self.sound.play()

    def pause(self):
        self.audio_pos = self.sound.get_pos()
        self.sound.stop()

    def stop(self):
        self.audio_pos = 0
        self.sound.stop()
