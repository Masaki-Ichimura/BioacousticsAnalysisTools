from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty
from kivy.garden.matplotlib import FigureCanvasKivyAgg

import numpy as np
from kivy.uix.button import Button
import matplotlib.pyplot as plt

from app.gui.main_container import MainContainerWidget
from utils.audio import load_wav
from utils.plot import show_spec, show_wav

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_container.kv')


class EditContainerWidget(MainContainerWidget):
    audio_data = None
    audio_fs = NumericProperty(0)

    def select_button_clicked(self):
        sidebar = self.ids.edit_sidebar
        selected_node = sidebar.ids.audio_treeview.selected_node
        if selected_node and '.' in selected_node.text:
            audio_file = selected_node.text
            file_name = [fn for fn in sidebar.audio_files if audio_file in fn][0]

            self.audio_data, self.audio_fs = load_wav(file_name)

            working_container = self.ids.edit_working_container
            audio_timeline = working_container.ids.edit_audio_timeline

            fig, axes = plt.subplots(2, 1, tight_layout=True)
            show_wav(self.audio_data, self.audio_fs, ax=axes[0])
            axes[0].set_xlim(0, self.audio_data.shape[-1]/self.audio_fs)
            axes[0].tick_params(bottom=False, labelbottom=False)
            show_spec(self.audio_data, self.audio_fs, ax=axes[1])

            timeline_widget = FigureCanvasKivyAgg(fig)
            audio_timeline.ids.box.clear_widgets()
            audio_timeline.ids.box.add_widget(timeline_widget)

            fig.savefig('img.png')

class EditWorkingContainerWidget(MainContainerWidget):
    pass

class EditAudioTimelineWidget(MainContainerWidget):
    pass

class EditAudioDisplayWidget(MainContainerWidget):
    pass
