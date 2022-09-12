from kivy.lang import Builder
from kivy.properties import DictProperty

from batools.app.gui.widgets.sub_tab import SubTab

Builder.load_file(__file__[:-3]+'.kv')


class TargetTab(SubTab):
    audio_dict = DictProperty({})
