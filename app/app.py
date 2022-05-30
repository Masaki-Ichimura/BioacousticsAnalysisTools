from kivymd.app import MDApp
from kivy.uix.widget import Widget

import japanize_kivy


class Root(Widget):
    def init(self):
        main_menu = self.ids.main_menu

        self.init_edit_tab()
        self.init_offprocess_tab()

    def init_edit_tab(self):
        main_menu = self.ids.main_menu

        edit_container = main_menu.ids.edit.content

        working_container = edit_container.ids.working_container
        sidebar = edit_container.ids.sidebar

        audio_display = working_container.ids.audio_display
        audio_timeline = audio_display.ids.audio_timeline

        def touch_up_timeline(instance, event):
            if audio_timeline.audio_file and event.button == 'left':
                sound_length = audio_timeline.sound.length
                fig_width = instance.width

                audio_timeline.audio_pos = max(
                    0, min(event.pos[0], fig_width)/fig_width*sound_length
                )

        audio_timeline.ids.box_tl.bind(on_touch_up=touch_up_timeline)

    def init_offprocess_tab(self):
        main_menu = self.ids.main_menu

        offprocess_container = main_menu.ids.edit.content

class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'BioacousticsAnalysis'

    def build(self):
        self.theme_cls.theme_style = 'Dark'
        root = Root()
        root.init()
        return root
