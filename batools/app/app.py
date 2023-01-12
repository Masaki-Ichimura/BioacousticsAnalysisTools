import pathlib
import tempfile

from kivymd.app import MDApp
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.widget import Widget

import japanize_kivy

# load widgets
# .py
from batools.app.gui import (
    main_container, main_tabs,
    edit_workarea, edit_sidebar,
    offprocess_workarea, offprocess_sidebar,
    config_workarea
)
from batools.app.gui.widgets import audiodisplay, scrollable_treeview
from batools.app.gui.widgets.preprocess import preprocessed, silence_removal
from batools.app.gui.widgets.offprocess import general, frog
# .kv
Builder.load_file(str(pathlib.Path(__file__).parent/'gui'/'widgets'/'separator.kv'))


class Root(Widget):
    pass

class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'BioacousticsAnalysis'

    def on_start(self, *args, **kwargs):
        self.tmp_dir = tempfile.TemporaryDirectory()
        main_container = self.root_window.children[0].ids.main_container
        self.links = dict(
            edit_tab=main_container.ids.edit_tab,
            offprocess_tab=main_container.ids.offprocess_tab,
            config_tab=main_container.ids.config_tab
        )

        def on_drop_file(instance, value, x, y, *args):
            audio_path = value.decode()
            audio_dict = self.audio_path2dict(audio_path)

            current_tab = main_container.ids.tabbed_panel.current_tab

            if current_tab == self.links['edit_tab']:
                edit_sidebar = current_tab.ids.sidebar

                if audio_dict['label'] not in edit_sidebar.choosed_audio_labels:
                    edit_sidebar.choosed_audio_dicts.append(audio_dict)

            elif current_tab == self.links['offprocess_tab']:
                offprocess_sidebar = current_tab.ids.sidebar

                if audio_dict['label'] not in offprocess_sidebar.audio_labels:
                    offprocess_sidebar.audio_dicts.append(audio_dict)

        Window.bind(on_drop_file=on_drop_file)

    def audio_path2dict(self, audio_path):
        audio_label = audio_path.split('/')[-1]
        audio_label = audio_label[:-audio_label[::-1].index('.')-1]
        cache_dir = self.tmp_dir
        audio_cache = f'{cache_dir.name}/org_{audio_label}.wav'
        audio_data, audio_fs, audio_ch = None, None, -1

        audio_dict = dict(
            label=audio_label, path=audio_path, cache=audio_cache,
            data=audio_data, fs=audio_fs, ch=audio_ch
        )
        return audio_dict

    def build(self):
        self.theme_cls.theme_style = 'Dark'

        LabelBase.register(name='H6', fn_regular='ipaexg.ttf')
        self.theme_cls.font_styles['H6'] = ['ipaexg', 20, False, 0.15]

        return Root()
