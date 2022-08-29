import torch
import datetime
import matplotlib.pyplot as plt

from kivy.app import App
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
    target_audio_dicts = ListProperty([])
    target_audio_labels = ObjectProperty(set())

    def on_kv_post(self, *args, **kwargs):
        self.offprocess_container = self.parent.parent

    def clear_treeview(self):
        audio_treeview = self.ids.target_audio_treeview

        _ = [
            audio_treeview.remove_node(node)
            for node in list(audio_treeview.iterate_all_nodes())
        ]

    def on_target_audio_dicts(self, instance, value):
        audio_labels = set([ad['label'] for ad in self.target_audio_dicts])
        if self.target_audio_labels != audio_labels:
            self.target_audio_labels = audio_labels

    def on_target_audio_labels(self, instance, value):
        audio_treeview = self.ids.target_audio_treeview
        audio_dicts = self.target_audio_dicts

        self.clear_treeview()

        for ad in audio_dicts:
            audio_label, audio_data, audio_fs = ad['label'], ad['data'], ad['fs']
            metadata = {
                '再生時間': datetime.timedelta(seconds=audio_data.size(-1)//audio_fs),
                'オーディオチャンネル': audio_data.size(0),
                'サンプルレート': audio_fs
            }

            audio_node = audio_treeview.add_node(AudioTreeViewLabel(text=audio_label))
            _ = [
                audio_treeview.add_node(
                    AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node
                )
                for k, v in metadata.items()
            ]

    def select_button_clicked(self):
        selected_node = self.ids.target_audio_treeview.selected_node

        if selected_node:
            target_labels = [ad['label'] for ad in self.target_audio_dicts]
            if selected_node.text in target_labels:
                idx = target_labels.index(selected_node.text)
                audio_dict = self.target_audio_dicts[idx]

                working_container = self.offprocess_container.ids.working_container
                working_container.audio_dict = audio_dict
