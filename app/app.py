from kivymd.app import MDApp
from kivy.uix.widget import Widget

import japanize_kivy


class Root(Widget):
    pass

class MainApp(MDApp):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'BioacousticsAnalysis'

    def build(self):
        return Root()
