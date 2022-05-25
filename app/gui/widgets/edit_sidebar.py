import datetime
import matplotlib.pyplot as plt

from kivy.uix.widget import Widget
from kivy.uix.treeview import TreeViewLabel
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty, NumericProperty

from app.gui.widgets.sidebar import SidebarWidget
from app.gui.widgets.filechooser import FilechooserPopup
from utils.audio import metadata_wav, load_wav
from utils.plot import show_spec, show_wav


Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_sidebar.kv')


class AudioTreeViewLabel(TreeViewLabel):
    pass


class EditSidebarWidget(SidebarWidget):
    file_path = StringProperty('')
    filechooser_popup = ObjectProperty(None)

    choosed_audio_files = []
    target_audio_files = []

    audio_file = StringProperty('')
    audio_data = None
    audio_fs = NumericProperty(0)

    def choose_button_clicked(self):

        def choose(selections):
            self.filechooser_popup.dismiss()

            for selection in selections:
                file_path = str(selection)

                if file_path and file_path not in self.choosed_audio_files:
                    metadata = metadata_wav(file_path)
                    metadata = {
                        '再生時間': datetime.timedelta(
                            seconds=metadata['num_frames']//metadata['sample_rate']
                        ),
                        'オーディオチャンネル': metadata['num_channels'],
                        'サンプルレート': metadata['sample_rate'],
                        'ビット/サンプル': metadata['bits_per_sample'],
                    }

                    audio_treeview = self.ids.choosed_audio_treeview
                    audio_node = audio_treeview.add_node(
                        AudioTreeViewLabel(text=file_path.split('/')[-1])
                    )
                    _ = [
                        audio_treeview.add_node(
                            AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node
                        )
                        for k, v in metadata.items()
                    ]

                    self.choosed_audio_files.append(file_path)

        self.filechooser_popup = FilechooserPopup(load=choose)
        self.filechooser_popup.open()

    def move_button_clicked(self):
        self.target_audio_files.extend(self.choosed_audio_files)
        audio_treeview = self.ids.target_audio_treeview
        _ = [
            audio_treeview.add_node(AudioTreeViewLabel(text=node.text))
            for node in self.ids.choosed_audio_treeview.iterate_all_nodes()
            if node.level == 1
        ]

    def remove_button_clicked(self, mode='choosed'):
        if mode == 'choosed':
            audio_treeview = self.ids.choosed_audio_treeview
            audio_files = self.choosed_audio_files
        elif mode == 'target':
            audio_treeview = self.ids.target_audio_treeview
            audio_files = self.target_audio_files

        selected_node = audio_treeview.selected_node

        if selected_node:
            _ = [
                audio_files.remove(fn)
                for fn in audio_files if fn.split('/')[-1] == selected_node.text
            ]
            audio_treeview.remove_node(selected_node)

    def reset_button_clicked(self, mode='choosed'):
        if mode == 'choosed':
            audio_treeview = self.ids.choosed_audio_treeview
            audio_files = self.choosed_audio_files
        elif mode == 'target':
            audio_treeview = self.ids.target_audio_treeview
            audio_files = self.target_audio_files

        # 何故かイテレータのまま取り出すとノードが一つ残る(謎)ためリストに変換
        _ = [
            audio_treeview.remove_node(node)
            for node in list(audio_treeview.iterate_all_nodes())
        ]
        audio_files.clear()

    def select_button_clicked(self):
        selected_node = self.ids.choosed_audio_treeview.selected_node
        if selected_node and '.' in selected_node.text:
            audio_file = selected_node.text
            file_name = [
                fn for fn in self.choosed_audio_files
                if audio_file == fn.split('/')[-1]
            ][0]

            self.audio_file = file_name
            self.audio_data, self.audio_fs = load_wav(file_name)

            container = self.parent.parent

            working_container = container.ids.edit_working_container
            audio_display = working_container.ids.edit_audio_display
            audio_timeline = audio_display.ids.edit_audio_timeline
            audio_toolbar = audio_display.ids.edit_audio_toolbar

            fig, axes = plt.subplots(2, 1, tight_layout=True)
            show_wav(self.audio_data, self.audio_fs, ax=axes[0])
            axes[0].set_xlim(0, self.audio_data.shape[-1]/self.audio_fs)
            axes[0].tick_params(
                bottom=False, labelbottom=False, left=False, labelleft=False
            )
            show_spec(self.audio_data, self.audio_fs, ax=axes[1])
            axes[1].tick_params(
                bottom=False, labelbottom=False, left=False, labelleft=False
            )
            axes[1].xaxis.set_visible(False); axes[1].yaxis.set_visible(False)

            timeline_widget = FigureCanvasKivyAgg(fig)
            audio_timeline.ids.box.clear_widgets()
            audio_timeline.ids.box.add_widget(timeline_widget)

            # fig.savefig('img.png')

            audio_toolbar.audio_file = self.audio_file
            audio_toolbar.set_audio()
