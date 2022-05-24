from kivy.uix.widget import Widget
from kivy.uix.treeview import TreeViewLabel
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty

from app.gui.widgets.sidebar import SidebarWidget
from app.gui.widgets.filechooser import FilechooserPopup
from utils.audio import metadata_wav

import datetime

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_sidebar.kv')


class AudioMetadataLabel(TreeViewLabel):
    pass


class EditSidebarWidget(SidebarWidget):
    file_path = StringProperty('')
    filechooser_popup = ObjectProperty(None)

    audio_files = []

    def load_button_clicked(self):
        self.filechooser_popup = FilechooserPopup(load=self.load)
        self.filechooser_popup.open()

    def load(self, selections):
        self.filechooser_popup.dismiss()

        for selection in selections:
            file_path = str(selection)

            if file_path and file_path not in self.audio_files:
                metadata = metadata_wav(file_path)
                metadata = {
                    '再生時間': datetime.timedelta(
                        seconds=metadata['num_frames']//metadata['sample_rate']
                    ),
                    'オーディオチャンネル': metadata['num_channels'],
                    'サンプルレート': metadata['sample_rate'],
                    'ビット/サンプル': metadata['bits_per_sample'],
                }

                audio_node = self.ids.audio_treeview.add_node(
                    AudioMetadataLabel(text=file_path.split('/')[-1])
                )
                _ = [
                    self.ids.audio_treeview.add_node(
                        AudioMetadataLabel(text=f'{k}: {v}'), parent=audio_node
                    )
                    for k, v in metadata.items()
                ]

                self.audio_files.append(file_path)

    def reset_button_clicked(self):
        # 何故かイテレータのままだとノードが一つ残るためリストに変換
        _ = [
            self.ids.audio_treeview.remove_node(node)
            for node in list(self.ids.audio_treeview.iterate_all_nodes())
        ]
        self.audio_files = []
