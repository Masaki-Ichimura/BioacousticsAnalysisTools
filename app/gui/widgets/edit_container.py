from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty

from app.gui.main_container import MainContainerWidget

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_container.kv')


class EditContainerWidget(MainContainerWidget):
    pass

class EditWorkingContainerWidget(MainContainerWidget):
    pass

class EditAudioDisplayWidget(MainContainerWidget):
    pass

class EditAudioDetailWidget(MainContainerWidget):
    pass
