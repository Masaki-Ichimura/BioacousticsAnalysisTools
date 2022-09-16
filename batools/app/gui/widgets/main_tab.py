from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanelItem
from kivy.lang import Builder

Builder.load_file(__file__[:-3]+'.kv')


class MainTab(TabbedPanelItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
