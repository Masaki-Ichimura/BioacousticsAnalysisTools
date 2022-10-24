import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import DictProperty
from kivymd.uix.screen import MDScreen

from batools.app.gui.widgets.sub_tab import SubTab
from batools.utils.audio.bss.auxiva import AuxIVA
from batools.utils.audio.bss.fastmnmf import FastMNMF
from batools.utils.audio.bss.ilrma import ILRMA

Builder.load_file(__file__[:-3]+'.kv')


class GeneralTab(SubTab):
    audio_dict = DictProperty({})

    def on_kv_post(self, *args, **kwargs):
        self.ids.nav_drawer.set_state('open')

    def on_audio_dict(self, instance, value):
        separate = self.ids.separate
        separate.audio_dict = value

class GeneralSeparate(MDScreen):
    audio_dict = DictProperty({})
    mode = 'ilrma'

    def on_kv_post(self, *args, **kwargs):

        def on_current_active_segment(instance, value):
            if value:
                self.mode = value.text.lower()
                self.ids.screen_manager.current = self.mode

        self.ids.mode_control.bind(current_active_segment=on_current_active_segment)

        def on_disabled(instance, value):
            textfields = [
                self.ids.ilrma_n_src, self.ids.ilrma_n_iter, self.ids.ilrma_n_components,
                self.ids.auxiva_n_src, self.ids.auxiva_n_iter,
                self.ids.fastmnmf_n_src, self.ids.fastmnmf_n_iter, self.ids.fastmnmf_n_components
            ]
            for textfield in textfields:
                textfield.disabled = value

        self.ids.separate_button.bind(disabled=on_disabled)

    def on_audio_dict(self, instance, value):
        self.ids.separate_button.disabled = not (value and value['data'].size(0) > 1)
        self.init_separate_args()

    def get_func(self):
        app = App.get_running_app()
        config_tab = app.links['config_tab']
        stft_args = config_tab.ids.working_container.get_stft_args()

        if self.mode == 'ilrma':
            func = ILRMA(**stft_args)
        elif self.mode == 'auxiva':
            func = AuxIVA(**stft_args)
        elif self.mode == 'fastmnmf':
            func = FastMNMF(**stft_args)

        return func

    def init_separate_args(self):
        n_src = 1 if not self.audio_dict else self.audio_dict['data'].size(0)

        # ILRMA
        self.ids.ilrma_n_src.text = str(n_src)
        self.ids.ilrma_n_iter.text = '30'
        self.ids.ilrma_n_components.text = '4'

        # AuxIVA
        self.ids.auxiva_n_src.text = str(n_src)
        self.ids.auxiva_n_iter.text = '20'

        # FastMNMF
        self.ids.fastmnmf_n_src.text = str(n_src)
        self.ids.fastmnmf_n_iter.text = '30'
        self.ids.fastmnmf_n_components.text = '4'

    def get_separate_args(self):
        if self.mode == 'ilrma':
            args = dict(
                n_src=int(self.ids.ilrma_n_src.text),
                n_iter=int(self.ids.ilrma_n_iter.text),
                n_components=int(self.ids.ilrma_n_components.text)
            )
        elif self.mode == 'auxiva':
            args = dict(
                n_src=int(self.ids.auxiva_n_src.text),
                n_iter=int(self.ids.auxiva_n_iter.text)
            )
        elif self.mode == 'fastmnmf':
            args = dict(
                n_src=int(self.ids.fastmnmf_n_src.text),
                n_iter=int(self.ids.fastmnmf_n_iter.text),
                n_components=int(self.ids.fastmnmf_n_components.text)
            )

        return args

    def separate(self):
        if self.audio_dict:
            app = App.get_running_app()
            cache_dir = app.tmp_dir

            sep_fn = self.get_func()

            self.ids.progressbar.value = 0

            def check_progress(dt):
                n, total = sep_fn.pbar.n, sep_fn.pbar.total

                if sep_fn.pbar.disable:
                    self.ids.progressbar.value = 100
                    return False
                else:
                    self.ids.progressbar.value = 100 * n // total

            self.check_progress = Clock.schedule_interval(check_progress, .5)

            def separate_process():
                sep_data = sep_fn(self.audio_dict['data'], **self.get_separate_args())
                sep_fs = self.audio_dict['fs']
                sep_label = f'bss_{self.mode}_{self.audio_dict["label"]}'
                sep_cache = f'{cache_dir.name}/{sep_label}.wav'
                self.sep_dict = dict(
                    label=sep_label, path=None, cache=sep_cache, data=sep_data, fs=sep_fs, ch=-1
                )
                Clock.schedule_once(update_process)

            def update_process(dt):
                # self.parent_tab.ids.select.audio_dict = self.sep_dict
                # self.parent_tab.ids.screen_manager.current = 'select'
                self.ids.separate_button.disabled = False
                self.ids.mode_control.disabled = False
                self.ids.progressbar.value = 0

            thread = threading.Thread(target=separate_process)
            thread.start()

class GeneralLocalize(MDScreen):
    pass