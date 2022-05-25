import datetime
import matplotlib.pyplot as plt

from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty,  ListProperty, NumericProperty
from kivy.uix.widget import Widget
from kivy.uix.treeview import TreeViewLabel
from kivy.garden.matplotlib import FigureCanvasKivyAgg

from app.gui.widgets.sidebar import Sidebar
from app.gui.widgets.filechooser import FilechooserPopup
from utils.audio import metadata_wav, load_wav
from utils.plot import show_spec, show_wav

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/offprocess_sidebar.kv')


class AudioTreeViewLabel(TreeViewLabel):
    pass

class OffprocessSidebar(Sidebar):

    target_audio_files = ListProperty([])

    def set_target_treeview(self):
        audio_treeview = self.ids.target_audio_treeview
        _ = [
            audio_treeview.add_node(
                AudioTreeViewLabel(text=fn.split('/')[-1])
            )
            for fn in self.target_audio_files
        ]
