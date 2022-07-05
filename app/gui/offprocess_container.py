from kivy.lang import Builder
from kivy.uix.widget import Widget

from app.gui.widgets.container import Container

Builder.load_file(__file__[:-3]+'.kv')


class OffprocessContainer(Container):
    pass

class OffprocessWorkingContainer(Container):
    pass
