from kivymd.app import MDApp
from kivy.uix.widget import Widget

import japanize_kivy


class Root(Widget):
    def init(self):
        main_menu = self.ids.main_menu
        print(main_menu.ids.edit.parent)

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
            bar = instance.children[0].canvas.children[-1]
            audio_pos = audio_timeline.audio_pos
            fig_width = max(0, instance.width)
            bar.pos = (min(event.pos[0], fig_width), bar.pos[1])
            if audio_timeline.audio_file:
                sound = audio_timeline.sound
                audio_timeline.audio_pos = max(0, bar.pos[0]/fig_width*sound.length)

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
