from kivy.lang import Builder
from kivy.properties import StringProperty, ObjectProperty, NumericProperty, BooleanProperty, Clock
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

    check_pos = None
    check_dt = .1

    def set_audio(self):
        working_container = self.parent.parent.parent.parent
        self.audio_file = working_container.audio_file
        self.sound = SoundLoader.load(self.audio_file)

    def play(self):
        if self.sound.state == 'stop':
            self.sound.seek(self.audio_pos)
            self.sound.play()
            self.check_pos = Clock.schedule_interval(
                lambda dt: self.position(), self.check_dt
            )

    def pause(self):
        if self.sound.state == 'play':
            self.sound.stop()
            Clock.unschedule(self.check_pos)
            self.check_pos = None

    def stop(self):
        self.sound.stop()
        self.position(0)
        Clock.unschedule(self.check_pos)
        self.check_pos = None

    def position(self, pos=None):
        if pos is None:
            self.audio_pos = self.sound.get_pos()
        else:
            self.audio_pos = pos

        if self.check_pos:
            if self.sound.length - self.audio_pos <= 2*self.check_dt:
                self.audio_pos = 0
                return False
