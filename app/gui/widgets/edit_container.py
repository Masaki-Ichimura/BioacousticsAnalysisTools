from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty

from app.gui.main_container import MainContainer

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_container.kv')


class EditContainer(MainContainer):
    pass

class EditWorkingContainer(MainContainer):
    pass

class EditAudioDisplay(MainContainer):
    pass

class EditAudioDetail(MainContainer):
    pass
