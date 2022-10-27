from kivy.lang import Builder
from kivy.properties import DictProperty

from batools.app.gui.widgets.sub_tab import SubTab

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dialog import MDDialog
from kivy.metrics import dp


Builder.load_file(__file__[:-3]+'.kv')


class TargetTab(SubTab):
    audio_dict = DictProperty({})
    dialog = None

    def show_test_dialog(self):
        if not self.dialog:
            audio_display = TargetAudioDisplay()
            self.dialog = MDDialog(
                title='Figure',
                type="custom",
                content_cls=audio_display,
                size_hint=(None, None),
                size=(audio_display.width+dp(24)*2, audio_display.height)
            )

        self.dialog.content_cls.audio_dict = self.audio_dict
        self.dialog.open()

class TargetAudioDisplay(MDBoxLayout):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_timeline, audio_toolbar = self.ids.audio_timeline, self.ids.audio_toolbar
        audio_timeline.audio_dict = audio_toolbar.audio_dict = value