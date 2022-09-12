from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanelItem
from kivy.lang import Builder

Builder.load_file(__file__[:-3]+'.kv')


class MainTab(TabbedPanelItem):
    def on_kv_post(self, *args, **kwargs):
        app = App.get_running_app()
        self.app = app
