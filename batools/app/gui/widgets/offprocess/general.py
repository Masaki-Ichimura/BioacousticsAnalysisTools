import subprocess
import threading
from functools import partial

import torch
import matplotlib.pyplot as plt
from pyroomacoustics.beamforming import circular_2D_array
from pyroomacoustics.doa import detect_peaks

from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio.audio_ffpyplayer import SoundFFPy
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import DictProperty
from kivy.uix.treeview import TreeViewLabel
from kivy.utils import platform
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivymd.color_definitions import colors
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label.label import MDIcon
from kivymd.uix.screen import MDScreen
from kivymd.uix.selectioncontrol.selectioncontrol import MDCheckbox
from plyer import filechooser, notification

from batools.app.gui.widgets.sub_tab import SubTab
from batools.app.gui.widgets.audiodisplay import AudioMiniplot
from batools.utils.audio.bss.auxiva import AuxIVA
from batools.utils.audio.bss.fastmnmf import FastMNMF
from batools.utils.audio.bss.ilrma import ILRMA
from batools.utils.audio.doa.music import MUSIC
from batools.utils.audio.wave import save_wave

Builder.load_file(__file__[:-3]+'.kv')


class GeneralTab(SubTab):
    audio_dict = DictProperty({})

    def on_kv_post(self, *args, **kwargs):
        self.ids.nav_drawer.set_state('open')

    def on_audio_dict(self, instance, value):
        separate, localize = self.ids.separate, self.ids.localize
        separate.audio_dict = localize.audio_dict = value

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
        #self.ids.ilrma_wpe.state = 'normal'
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
                # wpe=True if self.ids.ilrma_wpe.state == 'down' else False,
                n_src=int(self.ids.ilrma_n_src.text),
                n_iter=int(self.ids.ilrma_n_iter.text),
                n_components=int(self.ids.ilrma_n_components.text),
                return_scms=True
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
                mic_index=0,
                accelerate=True,
                return_scms=True
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
                    if total:
                        self.ids.progressbar.value = 100 * n // total

            self.check_progress = Clock.schedule_interval(check_progress, .5)

            def separate_process():
                ret_val = sep_fn(self.audio_dict['data'], **self.get_separate_args())

                if type(ret_val) is dict:
                    sep_data = ret_val['signals']
                    scms = ret_val.get('scms')
                else:
                    sep_data = ret_val
                    scms = None

                self.parent_tab.ids.localize.scms = scms

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
                parent_tab = self.parent_tab
                parent_tab.ids.sepout.audio_dict = parent_tab.ids.localize.sep_dict = self.sep_dict
                if parent_tab.ids.screen_manager.current == 'separate':
                    parent_tab.ids.screen_manager.current = 'sepout'
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
                type='custom',
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
    audio_dict = DictProperty({})
    sep_dict = DictProperty({})
    scms = None
    mic_plots = None
    sound = None

    def on_kv_post(self, *args, **kwargs):
        self.fig_doa = plt.figure()

        self.fig_doa.patch.set_alpha(0)
        self.ids.box_doa.add_widget(FigureCanvasKivyAgg(self.fig_doa))

    def on_audio_dict(self, instance, value):
        if value:
            self.ids.ssl_freq_min.text = f'{0}'
            self.ids.ssl_freq_max.text = f'{self.audio_dict["fs"]//2}'
        self.ids.localize_by_target_signal.disabled = not value
        self.sep_dict = {}
        self.scms = None
        self.mic_plots = None

    def on_sep_dict(self, instance, value):
        self.ids.localize_by_separated_signal.disabled = self.ids.localize_by_selected_signal.disabled = not value

    def get_ssl_args(self):
        app = App.get_running_app()
        config_tab = app.links['config_tab']
        ssl_args = config_tab.ids.working_container.get_ssl_args()

        audio_dict = self.audio_dict
        ch, fs = audio_dict['data'].size(0), audio_dict['fs']

        args = dict(
            sample_rate=fs,
            n_fft=ssl_args['n_fft'],
            r=float(self.ids.ssl_r.text),
            c=float(self.ids.ssl_c.text),
            n_grid=int(self.ids.ssl_n_grid.text),
            n_src=int(self.ids.ssl_n_src.text),
            freq_range=list(map(float, [self.ids.ssl_freq_min.text, self.ids.ssl_freq_max.text])),
            frequency_normalization=True
        )

        if ssl_args.get('mic_array') == 'circ':
            args['mic_locs'] = circular_2D_array([0., 0.], ch, 0., ssl_args['circ_radius'])

        return args

    def plot_doa(self, Pbr, n_src=1, mic_locs=None):
        fig_doa = self.fig_doa
        fig_doa.clear()

        ax = fig_doa.add_subplot(projection='polar')

        B, R = Pbr.size()
        thetas = torch.arange(R+1)*(2*torch.pi/R)

        self.mic_plots = None
        if mic_locs is not None:
            mic_max_dist = 0.2

            mic_locs_polar = torch.view_as_complex(torch.from_numpy(mic_locs.T).contiguous())
            mic_locs_theta = mic_locs_polar.angle()
            mic_locs_abs = mic_locs_polar.abs()
            mic_locs_abs *= (mic_max_dist/mic_locs_abs.max())

            mic_plots = [ax.scatter(mic_locs_theta, mic_locs_abs, marker='x', s=15, color='black')]
            mic_plots.extend([
                ax.annotate(f'{i:02d}', (mic_locs_theta[i], mic_locs_abs[i]), size=8)
                for i in range(mic_locs_polar.size(0))
            ])
            self.mic_plots = mic_plots

        qbn, Qbn = [], []
        for Pr in Pbr:
            qn = torch.from_numpy(detect_peaks(Pr))
            qbn.append(qn[Pr[qn].argsort()[-n_src:]])
        qbn = torch.stack(qbn)
        Qbn = Pbr[torch.arange(B)[:, None], qbn]

        _ = [ax.plot(thetas, Pr, label=i) for i, Pr in enumerate(torch.cat([Pbr, Pbr[:, [0]]], dim=1))]

        colors = []
        for line in ax.get_lines():
            c = line.get_color()
            _ = [colors.append(c) for _ in range(n_src)]
        ax.scatter(thetas[qbn.view(-1)], Qbn.view(-1), marker='*', c=colors)

        ax.set_ylim([0, 1])
        ax.legend()
        self.display_mic_locs()

        peak_degs = thetas[qbn].rad2deg()
        self.display_doa_treeview(peak_degs)

    def display_mic_locs(self):
        alpha = 1 if self.ids.ssl_display_mic_locs_checkbox.state == 'down' else 0

        if self.mic_plots is not None:
            _ = [mic_plot.set_alpha(alpha) for mic_plot in self.mic_plots]

        self.fig_doa.canvas.draw()

    def display_doa_treeview(self, peak_degs):
        doa_treeview = self.ids.doa_treeview

        _ = [
            doa_treeview.remove_node(node)
            for node in list(doa_treeview.iterate_all_nodes())
        ]

        for n, peaks in enumerate(peak_degs):
            doa_node = doa_treeview.add_node(TreeViewLabel(text=f'src{n}'))
            _ = [
                doa_treeview.add_node(
                    TreeViewLabel(text=f'peak{i}: {peak.item():.2f}°'), parent=doa_node
                )
                for i, peak in enumerate(peaks)
            ]

    def localize(self, mode):
        ssl_args = self.get_ssl_args()

        _ = [child.reset() for child in self.ids.box_signals.children]
        self.ids.box_signals.clear_widgets()

        if mode == 'target':
            music = MUSIC(**ssl_args)

            app = App.get_running_app()
            config_tab = app.links['config_tab']
            stft_args = config_tab.ids.working_container.get_stft_args()

            signals = torch.stft(
                self.audio_dict['data'].type(torch.float64), return_complex=True, **stft_args
            ).permute(2, 1, 0)[None]

            Pbr = music(signals=signals)

            self.plot_doa(Pbr, n_src=ssl_args['n_src'], mic_locs=ssl_args['mic_locs'])
        else:
            ssl_args['n_src'] = 1
            music = MUSIC(**ssl_args)

            scms = self.scms
            sep_checkboxes = self.parent_tab.ids.sepout.checkboxes
            sep_miniplots = self.parent_tab.ids.sepout.ids.stack_sep.children[::-1]

            if mode == 'selected':
                selected_indices = [checkbox.active for checkbox in sep_checkboxes]
            else:
                selected_indices = [True] * scms.size(0)

            if any(selected_indices):
                Pbr = music(scms=scms[selected_indices])

                self.plot_doa(Pbr, n_src=1, mic_locs=ssl_args['mic_locs'])

                if self.sound is None:
                    self.sound = self.parent_tab.ids.sepout.sound

                ssl_miniplots = [
                    AudioMiniplot(
                        data=mp.audio_data, fs=mp.audio_fs, path=mp.audio_path,
                        sound=self.sound, size=mp.size, size_hint=(None, None)
                    )
                    for mp, selected in zip(sep_miniplots, selected_indices)
                    if selected
                ]
                for i, mp in enumerate(ssl_miniplots):
                    i_widget = MDIcon(
                        icon=f'numeric-{i}-box',
                        theme_text_color='Custom', text_color=colors['Blue']['A400']
                    )
                    i_widget.pos_hint = {'x': .05, 'y': .1}
                    i_widget.size_hint = (.25, None)
                    mp.add_widget(i_widget)
                    self.ids.box_signals.add_widget(mp)
