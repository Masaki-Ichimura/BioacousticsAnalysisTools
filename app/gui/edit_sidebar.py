import datetime

from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.treeview import TreeViewLabel
from kivy.properties import *

from app.gui.widgets.sidebar import Sidebar, AudioTreeViewLabel
# from app.gui.widgets.filetreevew import AudioTreeViewLabel
from app.gui.widgets.filechooser import FilechooserPopup
from utils.audio.wave import metadata_wave

Builder.load_file(__file__[:-3]+'.kv')


class EditSidebar(Sidebar):
    file_path = StringProperty('')
    filechooser_popup = ObjectProperty(None)

    choosed_audio_files = ListProperty([])
    target_audio_files = ListProperty([])

    def choose_button_clicked(self):

        def choose(selections):
            self.filechooser_popup.dismiss()

            audio_files = [
                selection
                for selection in selections
                if selection not in self.choosed_audio_files
            ]
            self.choosed_audio_files.extend(audio_files)

        self.filechooser_popup = FilechooserPopup(load=choose)
        self.filechooser_popup.open()

    def move_button_clicked(self):
        add_audio_files = [
            af for af in self.choosed_audio_files
            if af not in self.target_audio_files
        ]
        audio_treeview = self.ids.target_audio_treeview
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
            audio_files = [
                af
                for af in audio_files
                if (type(af) is dict and af['tag'] != selected_node.text)
                or (type(af) is str and af.split('/')[-1] != selected_node.text)
            ]

            if mode == 'choosed':
                self.choosed_audio_files = audio_files
            elif mode == 'target':
                self.target_audio_files = audio_files

    def reset_button_clicked(self, mode):
        if mode == 'choosed':
            audio_files = self.choosed_audio_files
        elif mode == 'target':
            audio_files = self.target_audio_files

        audio_files.clear()
        self.clear_treeview(mode)

    def select_button_clicked(self):
        selected_node = self.ids.choosed_audio_treeview.selected_node
        if selected_node and '.' in selected_node.text:
            audio_file = [
                fn for fn in self.choosed_audio_files
                if selected_node.text == fn.split('/')[-1]
            ][0]

            working_container = self.parent.parent.ids.working_container

            working_container.audio_file = audio_file

    def back_button_clicked(self):
        edit_container = self.parent.parent
        working_container = edit_container.ids.working_container
        audio_detail = working_container.ids.audio_detail
        preprocessed = audio_detail.ids.preprocessed
        preprocessed_files = preprocessed.audio_files

        self.target_audio_files.extend(preprocessed_files)

    def clear_treeview(self, mode):
        if mode == 'choosed':
            audio_treeview = self.ids.choosed_audio_treeview
        elif mode == 'target':
            audio_treeview = self.ids.target_audio_treeview

        _ = [
            audio_treeview.remove_node(node)
            for node in list(audio_treeview.iterate_all_nodes())
        ]

    def on_target_audio_files(self, instance, value):
        audio_treeview = self.ids.target_audio_treeview
        audio_files = self.target_audio_files

        main_menu = self.get_root_window().children[0].ids.main_menu
        offprocess_sidebar = main_menu.ids.offprocess_container.ids.sidebar
        offprocess_sidebar.target_audio_files = audio_files

        self.clear_treeview(mode='target')

        for af in audio_files:
            if type(af) is str:
                metadata = metadata_wave(af)
                metadata = {
                    '再生時間': datetime.timedelta(
                        seconds=metadata['num_frames']//metadata['sample_rate']
                    ),
                    'オーディオチャンネル': metadata['num_channels'],
                    'サンプルレート': metadata['sample_rate'],
                    'ビット/サンプル': metadata['bits_per_sample'],
                }
                tag = af.split('/')[-1]
            elif type(af) is dict:
                data, fs, tag = af['data'], af['fs'], af['tag']
                metadata = {
                    '再生時間': datetime.timedelta(seconds=data.size(-1)//fs),
                    'オーディオチャンネル': data.size(0),
                    'サンプルレート': fs
                }

            audio_node = audio_treeview.add_node(AudioTreeViewLabel(text=tag))
            _ = [
                audio_treeview.add_node(
                    AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node
                )
                for k, v in metadata.items()
            ]

    def on_choosed_audio_files(self, instance, value):
        audio_treeview = self.ids.choosed_audio_treeview
        audio_files = self.choosed_audio_files

        self.clear_treeview(mode='choosed')

        for af in audio_files:
            if type(af) is str:
                metadata = metadata_wave(af)
                metadata = {
                    '再生時間': datetime.timedelta(
                        seconds=metadata['num_frames']//metadata['sample_rate']
                    ),
                    'オーディオチャンネル': metadata['num_channels'],
                    'サンプルレート': metadata['sample_rate'],
                    'ビット/サンプル': metadata['bits_per_sample'],
                }
                tag = af.split('/')[-1]

            audio_node = audio_treeview.add_node(AudioTreeViewLabel(text=tag))
            _ = [
                audio_treeview.add_node(
                    AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node
                )
                for k, v in metadata.items()
            ]
