from kivy.lang import Builder
from kivy.properties import StringProperty, ObjectProperty, NumericProperty, BooleanProperty
from kivy.uix.widget import Widget
from kivy.core.audio import SoundLoader

from app.gui.main_container import MainContainer

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/audiotimeline.kv')


class AudioTimeline(MainContainer):
    pass

class AudioToolbar(MainContainer):
    audio_file = StringProperty('')
    sound = ObjectProperty(None)
    audio_pos = NumericProperty(0)
    pause_flag = BooleanProperty(False)

    def set_audio(self):
        working_container = self.parent.parent.parent.parent
        self.audio_file = working_container.audio_file
        self.sound = SoundLoader.load(self.audio_file)

    def play(self):
        if self.sound.state == 'stop':
            if self.pause_flag:
                self.pause_flag = False
            else:
                self.audio_pos = 0

            self.sound.seek(self.audio_pos)
            self.sound.play()

    def pause(self):
        if self.sound.state == 'play':
            self.audio_pos = self.sound.get_pos()
            self.sound.stop()
            self.pause_flag = True

    def stop(self):
        self.audio_pos = 0
        self.sound.stop()
