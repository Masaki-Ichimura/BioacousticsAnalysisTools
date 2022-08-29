import datetime

from kivy.lang import Builder
from kivy.properties import *

from app.gui.widgets.tab import Tab
from app.gui.widgets.sidebar import AudioTreeViewLabel

Builder.load_file(__file__[:-3]+'.kv')


class PreprocessedTab(Tab):
    audio_dicts = ListProperty([])
    audio_labels = ObjectProperty(set())

    def on_audio_dicts(self, instance, value):
        audio_labels = set([ad['label'] for ad in self.audio_dicts])
        if audio_labels != self.audio_labels:
            self.audio_labels = audio_labels

    def on_audio_labels(self, instance, value):
        audio_treeview = self.ids.audio_treeview

        for ad in self.audio_dicts:
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
