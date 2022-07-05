import datetime
import matplotlib.pyplot as plt

from kivy.lang import Builder
from kivy.properties import *
from kivy.uix.widget import Widget
from kivy.garden.matplotlib import FigureCanvasKivyAgg

from app.gui.widgets.sidebar import Sidebar, AudioTreeViewLabel
# from app.gui.widgets.filetreevew import AudioTreeViewLabel
from app.gui.widgets.filechooser import FilechooserPopup

Builder.load_file(__file__[:-3]+'.kv')


class OffprocessSidebar(Sidebar):
    target_audio_files = ListProperty([])

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
