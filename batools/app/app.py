import pathlib
import tempfile

from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.text import LabelBase
from kivy.uix.widget import Widget

import japanize_kivy

# load widgets
# .py
from batools.app.gui import (
    main_menu,
    edit_container, edit_sidebar, offprocess_container, offprocess_sidebar, config_container
)
from batools.app.gui.widgets import scrollable_treeview, audiodisplay
from batools.app.gui.widgets.preprocess import preprocessed, silence_removal
from batools.app.gui.widgets.offprocess import target, general, frog
# .kv
Builder.load_file(str(pathlib.Path(__file__).parent/'gui'/'widgets'/'separator.kv'))

Builder.load_file(str(pathlib.Path(__file__).parent/'main.kv'))


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
