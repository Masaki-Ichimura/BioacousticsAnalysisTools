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

class EditAudioDisplay(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_timeline, audio_toolbar = self.ids.audio_timeline, self.ids.audio_toolbar
        audio_timeline.audio_dict = audio_toolbar.audio_dict = value

class EditAudioDetail(Container):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        silence_removal, preprocessed = self.ids.silence_removal, self.ids.preprocessed
        silence_removal.audio_dict = value

        if value:
            fs_org = value['fs']
            resample, freqfilter_min, freqfilter_max = map(str, [fs_org, 0, fs_org//2])
        else:
            resample, freqfilter_min, freqfilter_max = '', '', ''

        preprocessed.ids.resample_value.text = resample
        preprocessed.ids.freqfilter_min_value.text = freqfilter_min
        preprocessed.ids.freqfilter_max_value.text = freqfilter_max

        _ = [
            setattr(getattr(preprocessed.ids, f'{name}_checkbox'), 'disabled', not value)
            for name in ['resample', 'freqfilter']
        ]
        _ = [
            setattr(getattr(preprocessed.ids, f'{name}_checkbox'), 'state', 'normal')
            for name in ['silence_removal', 'resample', 'freqfilter']
        ]
