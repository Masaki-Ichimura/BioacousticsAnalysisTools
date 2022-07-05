import datetime

from kivy.lang import Builder
from kivy.properties import *

from app.gui.widgets.tab import Tab
from app.gui.widgets.sidebar import AudioTreeViewLabel

Builder.load_file(__file__[:-3]+'.kv')


class PreprocessedTab(Tab):
    audio_files = ListProperty([])

    def on_audio_files(self, instance, value):
        audio_treeview = self.ids.audio_treeview
        for audio_file in self.audio_files:
            audio_data = audio_file['data']
            audio_fs = audio_file['fs']
            audio_tag = audio_file['tag']

            metadata = {
                '再生時間': datetime.timedelta(seconds=audio_data.size(-1)//audio_fs),
                'オーディオチャンネル': audio_data.size(0),
                'サンプルレート': audio_fs
            }

            audio_node = audio_treeview.add_node(
                AudioTreeViewLabel(text=audio_tag)
            )
            _ = [
                audio_treeview.add_node(
                    AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node
                )
                for k, v in metadata.items()
            ]
