import datetime

from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.treeview import TreeViewLabel
from kivy.properties import ObjectProperty, StringProperty, ListProperty,  NumericProperty

from app.gui.widgets.sidebar import Sidebar, AudioTreeViewLabel
from app.gui.widgets.filechooser import FilechooserPopup
from utils.audio import metadata_wav

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_sidebar.kv')


class EditSidebar(Sidebar):
    file_path = StringProperty('')
    filechooser_popup = ObjectProperty(None)

    choosed_audio_files = []
    target_audio_files = ListProperty([])

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
        add_audio_files = [
            fn for fn in self.choosed_audio_files
            if fn not in self.target_audio_files
        ]
        audio_treeview = self.ids.target_audio_treeview
        _ = [
            audio_treeview.add_node(AudioTreeViewLabel(text=node.text))
            for node in self.ids.choosed_audio_treeview.iterate_all_nodes()
            if node.level == 1 and any([node.text==fn.split('/')[-1] for fn in add_audio_files])
        ]
        self.target_audio_files.extend(add_audio_files)

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
            audio_file = [
                fn for fn in self.choosed_audio_files
                if selected_node.text == fn.split('/')[-1]
            ][0]

            working_container = self.parent.parent.ids.working_container

            working_container.audio_file = audio_file

    def on_target_audio_files(self, instance, value):
        main_menu = self.get_root_window().children[0].ids.main_menu
        offprocess_sidebar = main_menu.ids.offprocess_container.ids.sidebar
        offprocess_sidebar.target_audio_files = value

        offprocess_sidebar.set_target_treeview()
