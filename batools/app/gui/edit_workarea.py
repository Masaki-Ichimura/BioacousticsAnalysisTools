from kivy.lang import Builder
from kivy.properties import DictProperty

from batools.app.gui.widgets.container import Container
from batools.utils.audio.wave import load_wave

Builder.load_file(__file__[:-3]+'.kv')


class EditWorkingContainer(Container):
    audio_dict = DictProperty({})

    def on_kv_post(self, *args, **kwargs):
        audio_toolbar = self.ids.audio_display.ids.audio_toolbar
        audio_toolbar.root_audio_dict_container = self

    def on_audio_dict(self, instance, value):
        audio_dict = self.audio_dict
        audio_display, audio_detail = self.ids.audio_display, self.ids.audio_detail

        if audio_dict and audio_dict['data'] is None:
            audio_dict['data'], audio_dict['fs'] = load_wave(self.audio_dict['path'])

        audio_display.audio_dict = audio_detail.audio_dict = value

        preprocessed_tab = audio_detail.ids.preprocessed

        if audio_dict:
            fs_org = audio_dict['fs']
            resample_fs, freqfilter_fs_min, freqfilter_fs_max = map(str, [fs_org, 0, fs_org//2])
        else:
            resample_fs, freqfilter_fs_min, freqfilter_fs_max = '', '', ''

        preprocessed_tab.ids.resample_fs.text = resample_fs
        preprocessed_tab.ids.freqfilter_fs_min.text = freqfilter_fs_min
        preprocessed_tab.ids.freqfilter_fs_max.text = freqfilter_fs_max

        checkboxes = [
            getattr(preprocessed_tab.ids, f'{fn}_checkbox') for fn in ['resample', 'freqfilter']
        ]
        for checkbox in checkboxes:
            checkbox.disabled = not audio_dict

class EditAudioDisplay(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_timeline, audio_toolbar = self.ids.audio_timeline, self.ids.audio_toolbar
        audio_timeline.audio_dict = audio_toolbar.audio_dict = value

class EditAudioDetail(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        silence_removal = self.ids.silence_removal
        silence_removal.audio_dict = value
