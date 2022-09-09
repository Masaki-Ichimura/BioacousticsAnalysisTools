from kivy.lang import Builder
from kivy.properties import DictProperty

from batools.app.gui.widgets.tab import Tab

Builder.load_file(__file__[:-3]+'.kv')


class TargetTab(Tab):
    audio_dict = DictProperty({})
