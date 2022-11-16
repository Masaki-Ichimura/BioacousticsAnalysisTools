import subprocess
import threading
from functools import partial

from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio.audio_ffpyplayer import SoundFFPy
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import DictProperty
from kivy.utils import platform
from kivymd.color_definitions import colors
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.screen import MDScreen
from kivymd.uix.selectioncontrol.selectioncontrol import MDCheckbox
from plyer import filechooser, notification

from batools.app.gui.widgets.sub_tab import SubTab
from batools.app.gui.widgets.audiodisplay import AudioMiniplot
from batools.utils.audio.bss.auxiva import AuxIVA
from batools.utils.audio.bss.fastmnmf import FastMNMF
from batools.utils.audio.bss.ilrma import ILRMA
from batools.utils.audio.wave import save_wave

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
                self.ids.ilrma_n_iter, self.ids.ilrma_n_components,
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
        wpe = 'normal'

        # ILRMA
        self.ids.ilrma_wpe.state = 'normal'
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
                wpe=True if self.ids.ilrma_wpe.state == 'down' else False,
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
                n_components=int(self.ids.fastmnmf_n_components.text),
                initialize_ilrma=True
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

                config_tab = app.links['config_tab']
                if config_tab.ids.working_container.get_notify('separate'):
                    title = app.title
                    message = '音源分離プロセスが終了しました'

                    try:
                        notification.notify(title=title, message=message)
                    except Exception:
                        if platform == 'macosx':
                            sh = f'osascript -e \'display notification "{message}" with title "{title}" sound name "Crystal"\''
                            subprocess.run(sh, shell=True)


                Clock.schedule_once(update_process)

            def update_process(dt):
                self.parent_tab.ids.sepout.audio_dict = self.sep_dict
                self.parent_tab.ids.screen_manager.current = 'sepout'
                self.ids.separate_button.disabled = False
                self.ids.mode_control.disabled = False
                self.ids.progressbar.value = 0

            thread = threading.Thread(target=separate_process)
            thread.start()

class GeneralSepout(MDScreen):
    audio_dict = DictProperty({})
    checkboxes = []
    dialog = None
    sound = None

    def on_audio_dict(self, instance, value):
        if self.audio_dict:
            sep_data, sep_fs = self.audio_dict['data'], self.audio_dict['fs']
            sep_path = self.audio_dict['cache']
            dot_idx = -sep_path[::-1].index('.')-1
            ch_path = sep_path[:dot_idx]+'_ch{:02d}'+sep_path[dot_idx:]

            _ = [child.reset() for child in self.ids.stack_sep.children]
            self.ids.stack_sep.clear_widgets()

            checkboxes = []
            for ch, ch_data in enumerate(sep_data):
                save_wave(ch_path.format(ch), ch_data[None], sep_fs, normalization=True)

                if self.sound is None:
                    self.sound = SoundFFPy()

                audio_miniplot = AudioMiniplot(
                    data=ch_data, fs=sep_fs, path=ch_path.format(ch), size_hint=(1/3, 1/3),
                    sound=self.sound
                )

                checkbox_widget = MDCheckbox()
                checkbox_widget.selected_color = checkbox_widget.unselected_color = colors['Blue']['A400']
                checkbox_widget.pos_hint = {'x': .0, 'top': .4}
                checkbox_widget.size_hint = (.25, None)

                audio_dict = dict(
                    data=ch_data[None], fs=sep_fs, ch=-1, path=ch_path.format(ch), cache=None
                )

                press = lambda *args, **kwargs: self.show_audio_display(kwargs['audio_dict'])

                show_button_widget = MDIconButton(
                    icon='arrow-expand',
                    theme_icon_color='ContrastParentBackground',
                    on_press=partial(press, audio_dict=audio_dict),
                    pos_hint={'x': .75, 'top': 1.},
                    size_hint = (.25, None)
                )

                audio_miniplot.add_widget(checkbox_widget)
                audio_miniplot.add_widget(show_button_widget)
                checkboxes.append(checkbox_widget)

                self.ids.stack_sep.add_widget(audio_miniplot)

            self.checkboxes = checkboxes

    def show_audio_display(self, audio_dict):
        if self.dialog is None:
            audio_display = GeneralAudioDisplay()
            self.dialog = MDDialog(
                title='Figure',
                type="custom",
                content_cls=audio_display,
                size_hint=(None, None),
                size=(audio_display.width+dp(24)*2, audio_display.height)
            )

        self.dialog.content_cls.audio_dict = audio_dict
        self.dialog.open()

    def save(self):
        indices = [checkbox.active for checkbox in self.checkboxes]
        if self.audio_dict and any(indices):

            sct_data, sct_fs = self.audio_dict['data'][indices], self.audio_dict['fs']

            selections = filechooser.save_file(
                title='save selected audio file', filters=[('audio file', '*.wav')],
                use_extensions=True
            )

            if selections:
                sct_path = selections[0]

                app = App.get_running_app()
                config_tab = app.links['config_tab']
                save_args = config_tab.ids.working_container.get_save_args()

                if save_args['normalization']:
                    save_args['normalization'] = 'ch'

                save_wave(sct_path, sct_data, sct_fs, **save_args)

class GeneralAudioDisplay(MDBoxLayout):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_timeline, audio_toolbar = self.ids.audio_timeline, self.ids.audio_toolbar
        audio_timeline.audio_dict = audio_toolbar.audio_dict = value

class GeneralLocalize(MDScreen):
    pass