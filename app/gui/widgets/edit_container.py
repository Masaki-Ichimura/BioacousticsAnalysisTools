from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty

from app.gui.main_container import MainContainerWidget
from utils.audio import load_wav

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_container.kv')


class EditContainerWidget(MainContainerWidget):
    audio_data = None
    audio_fs = NumericProperty(0)

    def select_button_clicked(self):
        edit_sidebar = self.ids.edit_sidebar
        selected_node = edit_sidebar.ids.audio_treeview.selected_node
        if selected_node and '.' in selected_node.text:
            audio_file = selected_node.text
            file_name = [fn for fn in edit_sidebar.audio_files if audio_file in fn][0]

            self.audio_data, self.audio_fs = load_wav(file_name)

class EditWorkingContainerWidget(MainContainerWidget):
    pass

class EditAudioTimelineWidget(MainContainerWidget):
    pass

class EditAudioDisplayWidget(MainContainerWidget):
    pass
