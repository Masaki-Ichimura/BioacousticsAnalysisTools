import datetime
import gc
from torchaudio.sox_effects import apply_effects_tensor

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ListProperty, ObjectProperty

from batools.app.gui.widgets.sub_tab import SubTab
from batools.app.gui.widgets.scrollable_treeview import AudioTreeViewLabel

Builder.load_file(__file__[:-3]+'.kv')


class PreprocessedTab(SubTab):
    audio_dicts = ListProperty([])
    audio_labels = ObjectProperty(set())

    def on_audio_dicts(self, instance, value):
        if value:
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

    def ok_button_clicked(self):
        if self.ids.silence_removal_checkbox.state == 'down':
            silence_removal = self.parent_tab.ids.silence_removal
            audio_dicts = silence_removal.extract()
        else:
            app = App.get_running_app()
            audio_dicts = [app.links['edit_tab'].ids.working_container.audio_dict]

        if any(audio_dicts):
            effects = []
            fs_org = audio_dicts[0]['fs']

            if self.ids.resample_checkbox.state == 'down':
                if self.ids.resample_fs.text:
                    fs_new = int(self.ids.resample_fs.text)
                    fs_new = fs_org if fs_new < 0 else fs_new
                else:
                    fs_new = fs_org

                if fs_new != fs_org:
                    effects.append(['rate', str(fs_new)])

            if self.ids.freqfilter_checkbox.state == 'down':
                if self.ids.freqfilter_fs_min.text:
                    freqfilter_fs_min = min(max(int(self.ids.freqfilter_fs_min.text), 0), fs_org//2)
                else:
                    freqfilter_fs_min = 0

                if self.ids.freqfilter_fs_max.text:
                    freqfilter_fs_max = max(min(int(self.ids.freqfilter_fs_max.text), fs_org//2), 0)
                else:
                    freqfilter_fs_max = fs_org//2

                if freqfilter_fs_min < freqfilter_fs_max:
                    freqfilter_fs_min = freqfilter_fs_min if freqfilter_fs_min != 0 else ''
                    freqfilter_fs_max = freqfilter_fs_max if freqfilter_fs_max != fs_org//2 else ''

                    if not freqfilter_fs_max:
                        sinc_arg = f'{freqfilter_fs_min}'
                    else:
                        sinc_arg = f'{freqfilter_fs_min}-{freqfilter_fs_max}'

                    if sinc_arg:
                        effects.append(['sinc', '-n 32767', sinc_arg])

            if effects:
                _ = [
                    audio_dict.update(dict(zip(
                        ('data', 'fs'),
                        apply_effects_tensor(
                            audio_dict['data'], audio_dict['fs'], effects, channels_first=True
                        )
                    ))) for audio_dict in audio_dicts
                ]

            self.audio_dicts.extend(audio_dicts)
