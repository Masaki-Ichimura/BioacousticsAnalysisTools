import tempfile

from kivymd.app import MDApp
from kivy.uix.widget import Widget

# Kivy を読み込んだ後に読み込むこと
import japanize_kivy


class Root(Widget):
    def on_kv_post(self, *args, **kwargs):
        self.tmp_dir = tempfile.TemporaryDirectory()


class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'BioacousticsAnalysis'

    def build(self):
        self.theme_cls.theme_style = 'Dark'
        return Root()
