import pathlib
import tempfile

from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.text import LabelBase
from kivy.uix.widget import Widget

# Kivy を読み込んだ後に読み込むこと
import japanize_kivy

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/main.kv')

class Root(Widget):
    pass

class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'BioacousticsAnalysis'

    def on_start(self, *args, **kwargs):
        self.tmp_dir = tempfile.TemporaryDirectory()
        main_menu = self.root_window.children[0].ids.main_menu
        self.links = dict(
            edit_container=main_menu.ids.edit_container,
            offprocess_container=main_menu.ids.offprocess_container,
            config_container=main_menu.ids.config_container
        )

    def build(self):
        self.theme_cls.theme_style = 'Dark'

        LabelBase.register(name='H6', fn_regular='ipaexg.ttf')
        self.theme_cls.font_styles['H6'] = ['ipaexg', 20, False, 0.15]

        return Root()
