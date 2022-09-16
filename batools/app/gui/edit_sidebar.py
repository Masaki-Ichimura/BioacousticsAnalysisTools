import datetime
import gc

from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty

from batools.app.gui.widgets.sidebar import Sidebar
from batools.app.gui.widgets.scrollable_treeview import AudioTreeViewLabel
from batools.app.gui.widgets.filechooser import FilechooserPopup
from batools.utils.audio.wave import metadata_wave, load_wave

Builder.load_file(__file__[:-3]+'.kv')


class EditSidebar(Sidebar):
    filechooser_popup = ObjectProperty(None)

    choosed_audio_dicts = ListProperty([])
    choosed_audio_labels = ObjectProperty(set())
    target_audio_dicts = ListProperty([])
    target_audio_labels = ObjectProperty(set())

    def choose_button_clicked(self):
        cache_dir = self.parent_tab.app.tmp_dir

        def choose(selections):
            self.filechooser_popup.dismiss()

            choosed_labels = [ad['label'] for ad in self.choosed_audio_dicts]
            add_dicts = []
            for selection in selections:
                audio_label = selection.split('/')[-1]
                audio_path, audio_cache = selection, f'{cache_dir.name}/org_{audio_label}.wav'
                audio_data, audio_fs, audio_ch = None, None, -1

                if audio_label not in choosed_labels:
                    add_dicts.append(dict(
                        label=audio_label,
                        path=audio_path, cache=audio_cache,
                        data=audio_data, fs=audio_fs, ch=audio_ch
                    ))

            self.choosed_audio_dicts.extend(add_dicts)

        self.filechooser_popup = FilechooserPopup(load=choose)
        self.filechooser_popup.open()

    def move_button_clicked(self):
        audio_dicts = self.choosed_audio_dicts
        selected_node = self.ids.choosed_audio_treeview.selected_node

        if selected_node:
            selected_label = selected_node.text
            audio_labels = [ad['label'] for ad in audio_dicts]

            if selected_label in audio_labels and selected_label not in self.target_audio_labels:
                self.target_audio_dicts.extend([audio_dicts[audio_labels.index(selected_label)]])

    def move_full_button_clicked(self):
        audio_dicts = self.choosed_audio_dicts

        self.target_audio_dicts.extend(
            [ad for ad in audio_dicts if ad['label'] not in self.target_audio_labels]
        )

    def remove_button_clicked(self, mode):
        if mode == 'choosed':
            audio_treeview, audio_dicts = self.ids.choosed_audio_treeview, self.choosed_audio_dicts
        elif mode == 'target':
            audio_treeview, audio_dicts = self.ids.target_audio_treeview, self.target_audio_dicts

        selected_node = audio_treeview.selected_node

        if selected_node:
            selected_label = selected_node.text
            audio_labels = [ad['label'] for ad in audio_dicts]

            if selected_label in audio_labels:
                audio_dicts.pop(audio_labels.index(selected_label))

    def reset_button_clicked(self, mode):
        if mode == 'choosed':
            audio_dicts, audio_labels = self.choosed_audio_dicts, self.choosed_audio_labels
        elif mode == 'target':
            audio_dicts, audio_labels = self.target_audio_dicts, self.target_audio_labels

        _ = [elem.clear() for elem in [audio_dicts, audio_labels]]

        self.clear_treeview(mode)
        gc.collect()

    def sort_button_clicked(self, mode):
        if mode == 'choosed':
            audio_dicts = self.choosed_audio_dicts
        elif mode == 'target':
            audio_dicts = self.target_audio_dicts

        audio_dicts.sort(key=lambda x: x['label'])

        self.add_treeview(mode)

    def select_button_clicked(self):
        audio_dicts = self.choosed_audio_dicts
        selected_node = self.ids.choosed_audio_treeview.selected_node

        if selected_node:
            selected_label = selected_node.text
            audio_labels = [ad['label'] for ad in audio_dicts]

            if selected_label in audio_labels:
                audio_dict = audio_dicts[audio_labels.index(selected_label)]

                working_container = self.parent_tab.ids.working_container
                working_container.audio_dict = audio_dict

    def fetch_button_clicked(self):
        working_container = self.parent_tab.ids.working_container
        audio_detail = working_container.ids.audio_detail
        tabs = audio_detail.ids.tabs

        if tabs.get_current_tab().icon == 'format-list-bulleted':
            preprocessed = audio_detail.ids.preprocessed
            audio_dicts, audio_treeview = preprocessed.audio_dicts, preprocessed.ids.audio_treeview

            selected_node = audio_treeview.selected_node

            if selected_node:
                selected_label = selected_node.text
                audio_labels = [ad['label'] for ad in audio_dicts]

                if selected_label in audio_labels:
                    self.target_audio_dicts.extend([audio_dicts[audio_labels.index(selected_label)]])

    def fetch_full_button_clicked(self):
        working_container = self.parent_tab.ids.working_container
        audio_detail = working_container.ids.audio_detail
        tabs = audio_detail.ids.tabs

        if tabs.get_current_tab().icon == 'format-list-bulleted':
            preprocessed = audio_detail.ids.preprocessed
            audio_dicts = preprocessed.audio_dicts

            self.target_audio_dicts.extend(
                [ad for ad in audio_dicts if ad['label'] not in self.target_audio_labels]
            )

    def clear_treeview(self, mode):
        if mode == 'choosed':
            audio_treeview = self.ids.choosed_audio_treeview
        elif mode == 'target':
            audio_treeview = self.ids.target_audio_treeview

        _ = [
            audio_treeview.remove_node(node) for node in list(audio_treeview.iterate_all_nodes())
        ]

    def add_treeview(self, mode):
        self.clear_treeview(mode)

        if mode == 'choosed':
            audio_dicts, audio_treeview = self.choosed_audio_dicts, self.ids.choosed_audio_treeview

            for ad in audio_dicts:
                if ad['data'] is None:
                    ad['data'], ad['fs'] = load_wave(ad['path'])

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
        elif mode == 'target':
            audio_dicts, audio_treeview = self.target_audio_dicts, self.ids.target_audio_treeview

            for ad in audio_dicts:
                audio_label, audio_data, audio_fs = ad['label'], ad['data'], ad['fs']

                try:
                    metadata = metadata_wave(ad['path'])
                    metadata = {
                        '再生時間': datetime.timedelta(
                            seconds=metadata['num_frames']//metadata['sample_rate']
                        ),
                        'オーディオチャンネル': metadata['num_channels'],
                        'サンプルレート': metadata['sample_rate'],
                        'ビット/サンプル': metadata['bits_per_sample'],
                    }
                except RuntimeError:
                    metadata = {
                        '再生時間': datetime.timedelta(seconds=audio_data.size(-1)//audio_fs),
                        'オーディオチャンネル': audio_data.size(0),
                        'サンプルレート': audio_fs
                    }

                audio_node = audio_treeview.add_node(AudioTreeViewLabel(text=ad['label']))
                _ = [
                    audio_treeview.add_node(AudioTreeViewLabel(text=f'{k}: {v}'), parent=audio_node)
                    for k, v in metadata.items()
                ]

    def on_target_audio_dicts(self, instance, value):
        audio_labels = set([ad['label'] for ad in self.target_audio_dicts])

        if audio_labels != self.target_audio_labels:
            self.target_audio_labels = audio_labels

    def on_target_audio_labels(self, instance, value):
        self.add_treeview(mode='target')
        audio_dicts = self.target_audio_dicts

        main_container = self.get_root_window().children[0].ids.main_container
        offprocess_sidebar = main_container.ids.offprocess_tab.ids.sidebar
        offprocess_sidebar.audio_dicts = audio_dicts

    def on_choosed_audio_dicts(self, instance, value):
        audio_labels = set([ad['label'] for ad in self.choosed_audio_dicts])

        if audio_labels != self.choosed_audio_labels:
            self.choosed_audio_labels = audio_labels

    def on_choosed_audio_labels(self, instance, value):
        self.add_treeview(mode='choosed')
