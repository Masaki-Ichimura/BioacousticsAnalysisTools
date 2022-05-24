from kivy.app import App
from kivy.uix.widget import Widget

import japanize_kivy


class RootWidget(Widget):
    pass

class MainApp(App):
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = 'BioacousticsAnalysis'

    def build(self):
        return RootWidget()
