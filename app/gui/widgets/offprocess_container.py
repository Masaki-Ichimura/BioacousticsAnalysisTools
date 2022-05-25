from kivy.uix.widget import Widget
from kivy.lang import Builder

from app.gui.main_container import MainContainer

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/offprocess_container.kv')


class OffprocessContainer(MainContainer):
    pass

class OffprocessWorkingContainer(MainContainer):
    pass
