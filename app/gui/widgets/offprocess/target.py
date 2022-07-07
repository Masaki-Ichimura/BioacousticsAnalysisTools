from kivy.lang import Builder

from app.gui.widgets.container import Container
from app.gui.widgets.tab import Tab

Builder.load_file(__file__[:-3]+'.kv')


class TargetTab(Tab):
    target_signal = None

class TargetAudioDisplay(Container):
    pass
