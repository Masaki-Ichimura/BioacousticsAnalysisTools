import datetime
import gc
import pathlib

import torchaudio
from torchaudio.sox_effects import apply_effects_tensor

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty
from plyer import filechooser

from batools.app.gui.widgets.sub_tab import SubTab
from batools.app.gui.widgets.scrollable_treeview import AudioTreeViewLabel
from batools.utils.audio.wave import save_wave

Builder.load_file(__file__[:-3]+'.kv')


class PreprocessedTab(SubTab):
    audio_dicts = ListProperty([])
    audio_labels = ObjectProperty(set())

    def on_audio_dicts(self, instance, value):
        audio_labels = set([ad['label'] for ad in value])
        if audio_labels != self.audio_labels:
            self.audio_labels = audio_labels

    def on_audio_labels(self, instance, value):
        self.add_treeview()

    def remove_button_clicked(self, select):
        audio_dicts, audio_labels = self.audio_dicts, self.audio_labels

        if select:
            selected_node = self.ids.audio_treeview.selected_node

            if selected_node:
                selected_label = selected_node.text

                if selected_label in audio_labels:
                    audio_labels = [ad['label'] for ad in audio_dicts]
                    audio_dicts.pop(audio_labels.index(selected_label))
        else:
            _ = [elem.clear() for elem in [audio_dicts, audio_labels]]
            self.clear_treeview()
            gc.collect()

    def sort_button_clicked(self, value, button):
        sort_args = {}
        if value == 'label':
            sort_args['key'] = lambda x: x['label']
        elif value == 'duration':
            sort_args['key'] = lambda x: x['data'].size(-1)

        if button.icon.split('-')[-1] == 'ascending':
            sort_args['reverse'] = False
            button.icon = button.icon[:-button.icon[::-1].index('-')] + 'descending'
        elif button.icon.split('-')[-1] == 'descending':
            sort_args['reverse'] = True
            button.icon = button.icon[:-button.icon[::-1].index('-')] + 'ascending'

        if sort_args:
            self.audio_dicts.sort(**sort_args)
            self.add_treeview()

    def save_button_clicked(self, select):
        audio_dicts, audio_labels = self.audio_dicts, self.audio_labels

        app = App.get_running_app()
        config_tab = app.links['config_tab']
        save_args = config_tab.ids.working_container.get_save_args()

        if select:
            selected_node = self.ids.audio_treeview.selected_node

            if selected_node:
                selected_label = selected_node.text

                if selected_label in audio_labels:
                    audio_labels = [ad['label'] for ad in audio_dicts]
                    audio_dict = audio_dicts[audio_labels.index(selected_label)]
                else:
                    return None

                selections = filechooser.save_file(
                    title='save selected audio file', filters=[('audio file', '*.wav')],
                    use_extensions=True
                )

                if selections:
                    audio_path = selections[0]
                    audio_data, audio_fs = audio_dict['data'], audio_dict['fs']

                    save_wave(audio_path, audio_data, audio_fs, **save_args)

        else:
            if self.audio_labels:
                selections = filechooser.choose_dir(
                    title='save audio file', filters=[('audio file', '*.wav')],
                    use_extensions=True
                )

                if selections:
                    audio_dir = pathlib.Path(selections[0])
                    _ = [
                        save_wave(
                            str(audio_dir/f'{ad["label"]}.wav'), ad['data'], ad['fs'], **save_args
                        )
                        for ad in audio_dicts
                    ]

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

    def ok_button_clicked(self):
        if self.ids.silence_removal_checkbox.state == 'down':
            silence_removal = self.parent_tab.ids.silence_removal
            audio_dicts = silence_removal.extract(self.ids.label_value.text)
        else:
            app = App.get_running_app()
            audio_dict = app.links['edit_tab'].ids.working_container.audio_dict

            audio_dicts = [{
                k: v if k != 'label' else f'{v}_preprocessed'
                for k, v in audio_dict.items()
            }]

        if any(audio_dicts):
            effects = []
            fs_org = audio_dicts[0]['fs']

            if self.ids.resample_checkbox.state == 'down':
                if self.ids.resample_value.text:
                    fs_new = int(self.ids.resample_value.text)
                    fs_new = fs_org if fs_new < 0 else fs_new
                else:
                    fs_new = fs_org

                if fs_new != fs_org:
                    effects.append(['rate', str(fs_new)])

            if self.ids.freqfilter_checkbox.state == 'down':
                if self.ids.freqfilter_min_value.text:
                    freqfilter_min = min(max(int(self.ids.freqfilter_min_value.text), 0), fs_org//2)
                else:
                    freqfilter_min = 0

                if self.ids.freqfilter_max_value.text:
                    freqfilter_max = max(min(int(self.ids.freqfilter_max_value.text), fs_org//2), 0)
                else:
                    freqfilter_max = fs_org//2

                if freqfilter_min < freqfilter_max:
                    freqfilter_min = freqfilter_min if freqfilter_min != 0 else ''
                    freqfilter_max = freqfilter_max if freqfilter_max != fs_org//2 else ''

                    if not freqfilter_max:
                        sinc_arg = f'{freqfilter_min}'
                    else:
                        sinc_arg = f'{freqfilter_min}-{freqfilter_max}'

                    if sinc_arg:
                        effects.append(['sinc', '-n 32767', sinc_arg])

            if effects:
                _ = [
                    ad.update(dict(zip(
                        ('data', 'fs'),
                        apply_effects_tensor(ad['data'], ad['fs'], effects, channels_first=True)
                    ))) for ad in audio_dicts
                ]

            audio_labels = [ad['label'] for ad in audio_dicts]
            add_dicts = []
            for ad in self.audio_dicts:
                if ad['label'] in audio_labels:
                    idx = audio_labels.index(ad['label'])
                    add_dicts.append(audio_dicts[idx])
                    _ = [elem.pop(idx) for elem in [audio_dicts, audio_labels]]
                else:
                    add_dicts.append(ad)

            add_dicts += audio_dicts

            self.audio_dicts = add_dicts
            self.add_treeview()
