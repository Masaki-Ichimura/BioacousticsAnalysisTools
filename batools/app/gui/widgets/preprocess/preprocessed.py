import datetime
import gc

from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty

from batools.app.gui.widgets.sub_tab import SubTab
from batools.app.gui.widgets.scrollable_treeview import AudioTreeViewLabel

Builder.load_file(__file__[:-3]+'.kv')


class PreprocessedTab(SubTab):
    audio_dicts = ListProperty([])
    audio_labels = ObjectProperty(set())

    def on_audio_dicts(self, instance, value):
        audio_labels = set([ad['label'] for ad in self.audio_dicts])
        if audio_labels != self.audio_labels:
            self.audio_labels = audio_labels

    def on_audio_labels(self, instance, value):
        self.add_treeview()

    def remove_button_clicked(self):
        audio_treeview = self.ids.audio_treeview
        audio_dicts = self.audio_dicts

        selected_node = audio_treeview.selected_node

        if selected_node:
            selected_label = selected_node.text
            audio_labels = [ad['label'] for ad in audio_dicts]

            if selected_label in audio_labels:
                audio_dicts.pop(audio_labels.index(selected_label))

    def reset_button_clicked(self):
        audio_dicts, audio_labels = self.audio_dicts, self.audio_labels

        _ = [elem.clear() for elem in [audio_dicts, audio_labels]]
        self.clear_treeview()
        gc.collect()

    def sort_button_clicked(self):
        self.audio_dicts.sort(key=lambda x: x['label'])

        self.add_treeview()

    def clear_treeview(self):
        audio_treeview = self.ids.audio_treeview

        _ = [
            audio_treeview.remove_node(node) for node in list(audio_treeview.iterate_all_nodes())
        ]

    def add_treeview(self):
        self.clear_treeview()

        audio_dicts, audio_treeview = self.audio_dicts, self.ids.audio_treeview

        for ad in audio_dicts:
             audio_label, audio_data, audio_fs = ad['label'], ad['data'], ad['fs']
             metadata = {
                 '再生時間': datetime.timedelta(seconds=audio_data.size(-1)//audio_fs),
                 'オーディオチャンネル': audio_data.size(0),
                 'サンプルレート': audio_fs
             }

             audio_node = audio_treeview.add_node(AudioTreeViewLabel(text=audio_label))
             _ = [
                 audio_treeview.add_node(AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node)
                 for k, v in metadata.items()
             ]
