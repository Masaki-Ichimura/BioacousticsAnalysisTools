from kivy.lang import Builder
from kivy.uix.widget import Widget

from app.gui.widgets.container import Container

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/offprocess_container.kv')


class OffprocessContainer(Container):
    pass

class OffprocessWorkingContainer(Container):
    pass
