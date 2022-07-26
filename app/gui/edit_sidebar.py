import datetime

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.treeview import TreeViewLabel
from kivy.properties import *

from app.gui.widgets.sidebar import Sidebar, AudioTreeViewLabel
# from app.gui.widgets.filetreevew import AudioTreeViewLabel
from app.gui.widgets.filechooser import FilechooserPopup
from utils.audio.wave import metadata_wave, load_wave

Builder.load_file(__file__[:-3]+'.kv')


class EditSidebar(Sidebar):
    filechooser_popup = ObjectProperty(None)

    choosed_audio_dicts = ListProperty([])
    choosed_audio_tags = ObjectProperty(set())
    target_audio_dicts = ListProperty([])
    target_audio_tags = ObjectProperty(set())

    def on_kv_post(self, *args, **kwargs):
        self.edit_container = self.parent.parent

    def choose_button_clicked(self):
        cache_dir = self.edit_container.app.tmp_dir

        def choose(selections):
            self.filechooser_popup.dismiss()

            choosed_tags = [ad['tag'] for ad in self.choosed_audio_dicts]
            add_dicts = []
            for selection in selections:
                audio_tag = selection.split('/')[-1]
                audio_path, audio_cache = selection, f'{cache_dir.name}/org_{audio_tag}.wav'
                audio_data, audio_fs, audio_ch = None, None, -1

                if audio_tag not in choosed_tags:
                    add_dicts.append(dict(
                        tag=audio_tag,
                        path=audio_path, cache=audio_cache,
                        data=audio_data, fs=audio_fs, ch=audio_ch
                    ))

            self.choosed_audio_dicts.extend(add_dicts)

        self.filechooser_popup = FilechooserPopup(load=choose)
        self.filechooser_popup.open()

    def move_button_clicked(self):
        target_tags = [ad['tag'] for ad in self.target_audio_dicts]
        add_dicts = [
            ad for ad in self.choosed_audio_dicts if ad['tag'] not in target_tags
        ]
        self.target_audio_dicts.extend(add_dicts)

    def remove_button_clicked(self, mode):
        if mode == 'choosed':
            audio_treeview = self.ids.choosed_audio_treeview
            audio_dicts = self.choosed_audio_dicts
        elif mode == 'target':
            audio_treeview = self.ids.target_audio_treeview
            audio_dicts = self.target_audio_dicts

        selected_node = audio_treeview.selected_node

        if selected_node:
            audio_tags = [ad['tag'] for ad in audio_dicts]
            if selected_node.text in audio_tags:
                idx = audio_tags.index(selected_node.text)
                audio_dicts.pop(idx)

    def reset_button_clicked(self, mode):
        if mode == 'choosed':
            audio_dicts = self.choosed_audio_dicts
        elif mode == 'target':
            audio_dicts = self.target_audio_dicts

        audio_dicts.clear()
        self.clear_treeview(mode)

    def select_button_clicked(self):
        selected_node = self.ids.choosed_audio_treeview.selected_node
        if selected_node:
            choosed_tags = [ad['tag'] for ad in self.choosed_audio_dicts]
            if selected_node.text in choosed_tags:
                idx = choosed_tags.index(selected_node.text)
                audio_dict = self.choosed_audio_dicts[idx]

                working_container = self.edit_container.ids.working_container
                working_container.audio_dict = audio_dict

    def fetch_button_clicked(self):
        working_container = self.edit_container.ids.working_container
        audio_detail = working_container.ids.audio_detail
        preprocessed = audio_detail.ids.preprocessed
        preprocessed_dicts = preprocessed.audio_dicts

        target_tags = [ad['tag'] for ad in self.target_audio_dicts]
        add_dicts = [
            ad for ad in preprocessed_dicts if ad['tag'] not in target_tags
        ]
        self.target_audio_dicts.extend(add_dicts)

    def clear_treeview(self, mode):
        if mode == 'choosed':
            audio_treeview = self.ids.choosed_audio_treeview
        elif mode == 'target':
            audio_treeview = self.ids.target_audio_treeview

        _ = [
            audio_treeview.remove_node(node)
            for node in list(audio_treeview.iterate_all_nodes())
        ]

    def on_target_audio_dicts(self, instance, value):
        audio_tags = set([ad['tag'] for ad in self.target_audio_dicts])
        if audio_tags != self.target_audio_tags:
            self.target_audio_tags = audio_tags

    def on_target_audio_tags(self, instance, value):
        audio_dicts = self.target_audio_dicts
        audio_treeview = self.ids.target_audio_treeview

        self.clear_treeview(mode='target')

        for ad in audio_dicts:
            if ad['data'] is None:
                ad['data'], ad['fs'] = load_wave(ad['path'])

            audio_tag, audio_data, audio_fs = ad['tag'], ad['data'], ad['fs']

            metadata = {
                '再生時間': datetime.timedelta(seconds=audio_data.size(-1)//audio_fs),
                'オーディオチャンネル': audio_data.size(0),
                'サンプルレート': audio_fs
            }

            audio_node = audio_treeview.add_node(AudioTreeViewLabel(text=audio_tag))
            _ = [
                audio_treeview.add_node(AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node)
                for k, v in metadata.items()
            ]

        main_menu = self.get_root_window().children[0].ids.main_menu
        offprocess_sidebar = main_menu.ids.offprocess_container.ids.sidebar
        offprocess_sidebar.target_audio_dicts = audio_dicts

    def on_choosed_audio_dicts(self, instance, value):
        audio_tags = set([ad['tag'] for ad in self.choosed_audio_dicts])
        if self.choosed_audio_tags != audio_tags:
            self.choosed_audio_tags = audio_tags

    def on_choosed_audio_tags(self, instance, value):
        audio_dicts = self.choosed_audio_dicts
        audio_treeview = self.ids.choosed_audio_treeview

        self.clear_treeview(mode='choosed')

        for ad in audio_dicts:
            metadata = metadata_wave(ad['path'])

            if ad['fs'] is None:
                ad['fs'] = metadata['sample_rate']

            metadata = {
                '再生時間': datetime.timedelta(
                    seconds=metadata['num_frames']//metadata['sample_rate']
                ),
                'オーディオチャンネル': metadata['num_channels'],
                'サンプルレート': metadata['sample_rate'],
                'ビット/サンプル': metadata['bits_per_sample'],
            }

            audio_node = audio_treeview.add_node(AudioTreeViewLabel(text=ad['tag']))
            _ = [
                audio_treeview.add_node(AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node)
                for k, v in metadata.items()
            ]
