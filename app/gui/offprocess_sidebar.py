import torch
import datetime
import matplotlib.pyplot as plt

from kivy.lang import Builder
from kivy.properties import *
from kivy.uix.widget import Widget
from kivy.garden.matplotlib import FigureCanvasKivyAgg

from app.gui.widgets.sidebar import Sidebar, AudioTreeViewLabel
from app.gui.widgets.filechooser import FilechooserPopup
from app.kivy_utils import TorchTensorProperty
from utils.audio.wave import load_wave

Builder.load_file(__file__[:-3]+'.kv')


class OffprocessSidebar(Sidebar):
    target_audio_files = ListProperty([])
    audio_data = TorchTensorProperty(torch.zeros(1))
    audio_monodata = TorchTensorProperty(torch.zeros(1))
    audio_fs = None

    def clear_treeview(self):
        audio_treeview = self.ids.target_audio_treeview

        _ = [
            audio_treeview.remove_node(node)
            for node in list(audio_treeview.iterate_all_nodes())
        ]

    def on_target_audio_files(self, instance, value):
        audio_treeview = self.ids.target_audio_treeview
        audio_files = self.target_audio_files

        if audio_files:
            self.clear_treeview()

            _ = [
                audio_treeview.add_node(AudioTreeViewLabel(text=af['tag']))
                if type(af) is dict
                else audio_treeview.add_node(AudioTreeViewLabel(text=af.split('/')[-1]))
                for af in audio_files
            ]

    def select_button_clicked(self):
        selected_node = self.ids.target_audio_treeview.selected_node

        if selected_node:
            audio_file = [
                af for af in self.target_audio_files
                if (type(af) is dict and af['tag'] == selected_node.text)
                or (type(af) is str and af.split('/')[-1] == selected_node.text)
            ]
            if audio_file:
                audio_file = audio_file[0]
                if type(audio_file) is dict:
                    audio_data, audio_fs = audio_file['data'], audio_file['fs']
                elif type(audio_file) is str:
                    audio_data, audio_fs = load_wave(audio_file)

                self.audio_fs = audio_fs
                self.audio_monodata = audio_data[0]

    def on_audio_monodata(self, instance, value):
        working_container = self.parent.parent.ids.working_container
        target = working_container.ids.target

        audio_timeline = target.ids.audio_display.ids.audio_timeline

        audio_timeline.audio_file = '-'
        audio_timeline.audio_fs = self.audio_fs
        audio_timeline.audio_data = self.audio_monodata[None]
