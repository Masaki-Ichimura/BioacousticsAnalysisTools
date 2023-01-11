import csv
import pathlib
import subprocess
import threading
from functools import partial
from itertools import combinations

import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio.audio_ffpyplayer import SoundFFPy
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import DictProperty
from kivy.uix.treeview import TreeViewLabel
from kivy.utils import platform
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from plyer import filechooser, notification
from kivymd.color_definitions import colors
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label.label import MDIcon
from kivymd.uix.screen import MDScreen
from kivymd.uix.selectioncontrol.selectioncontrol import MDCheckbox

from batools.app.gui.widgets.sub_tab import SubTab
from batools.app.gui.widgets.audiodisplay import AudioMiniplot
from batools.utils.audio.analysis.frog import check_synchronization
from batools.utils.audio.bss.auxiva import AuxIVA
from batools.utils.audio.bss.fastmnmf import FastMNMF
from batools.utils.audio.bss.ilrma import ILRMA
from batools.utils.audio.plot import show_wave
from batools.utils.audio.wave import save_wave

Builder.load_file(__file__[:-3]+'.kv')


class FrogTab(SubTab):
    audio_dict = DictProperty({})

    def on_kv_post(self, *args, **kwargs):
        self.ids.nav_drawer.set_state('open')

    def on_audio_dict(self, instance, value):
        separate = self.ids.separate
        separate.audio_dict = value

class FrogSeparate(MDScreen):
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

        # これをやってもスイッチの位置が更新されないが，一応残しておく
        # default_segment = [
        #     child for child in self.ids.mode_control.ids.segment_panel.children
        #     if type(child) is MDSegmentedControlItem and child.text.lower() == self.mode
        # ][0]
        # self.ids.mode_control.on_press_segment(default_segment, default_segment)

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
                n_components=int(self.ids.fastmnmf_n_components.text),
                mic_index=0,
                accelerate=True
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
                sep_data = sep_fn(self.audio_dict['data'], **self.get_separate_args())
                 # 尖度の大きい順に並び替え(小->雑音)
                indices_sorted_by_kurtosis = torch.tensor(
                    [kurtosis(ch_data) for ch_data in sep_data]
                ).sort(descending=True).indices
                sep_data = sep_data[indices_sorted_by_kurtosis]

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
                self.parent_tab.ids.select.audio_dict = self.sep_dict
                if self.parent_tab.ids.screen_manager.current == 'separate':
                    self.parent_tab.ids.screen_manager.current = 'select'
                self.ids.separate_button.disabled = False
                self.ids.mode_control.disabled = False
                self.ids.progressbar.value = 0

            thread = threading.Thread(target=separate_process)
            thread.start()

class FrogSelect(MDScreen):
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
            audio_display = FrogAudioDisplay()
            self.dialog = MDDialog(
                title='Figure',
                type='custom',
                content_cls=audio_display,
                size_hint=(None, None),
                size=(audio_display.width+dp(24)*2, audio_display.height)
            )

        self.dialog.content_cls.audio_dict = audio_dict
        self.dialog.open()

    def get_check_synchronization_args(self):
        args = dict(
            call_interval_ms=int(self.ids.call_interval_ms.text),
            minimum_amplitude_rate=float(self.ids.minimum_amplitude_rate.text)
        )
        return args

    def select(self):
        indices = [checkbox.active for checkbox in self.checkboxes]
        if self.audio_dict and sum(indices) > 1:
            app = App.get_running_app()
            cache_dir = app.tmp_dir

            sct_label = f'select_{self.audio_dict["label"]}'
            sct_data, sct_fs = self.audio_dict['data'][indices], self.audio_dict['fs']
            sct_cache = f'{cache_dir.name}/{sct_label}.wav'

            sct_dict = dict(
                label=sct_label, path=None, cache=sct_cache, data=sct_data, fs=sct_fs, ch=-1
            )

            self.parent_tab.ids.analysis.audio_dict = sct_dict

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

    def forward(self):
        if self.audio_dict:
            app = App.get_running_app()
            audio_detail = app.links['offprocess_tab'].ids.working_container.ids.audio_detail
            general = audio_detail.ids.general

            general.ids.sepout.audio_dict = self.audio_dict

class FrogAnalysis(MDScreen):
    audio_dict = DictProperty({})
    peaks = None
    results = None
    sound = None

    def on_kv_post(self, *args, **kwargs):
        self.fig_hist = plt.figure()
        self.treeview_callback = None

        self.fig_hist.patch.set_alpha(0)
        self.ids.box_hist.add_widget(FigureCanvasKivyAgg(self.fig_hist))

    def on_audio_dict(self, instance, value):
        if value:
            ana_datas, ana_fs = self.audio_dict['data'], self.audio_dict['fs']

            if ana_datas.size(0) > 1:
                combs = combinations(range(ana_datas.size(0)), 2)

                sep_miniplots = self.parent_tab.ids.select.ids.stack_sep.children
                sep_datas = [mp.audio_data for mp in sep_miniplots]

                sct_miniplots = [
                    sep_miniplots[[ana_data.equal(sep_data) for sep_data in sep_datas].index(True)]
                    for ana_data in ana_datas
                ]

                if self.sound is None:
                    self.sound = self.parent_tab.ids.select.sound

                sct_miniplots = [
                    AudioMiniplot(
                        data=mp.audio_data, fs=mp.audio_fs, path=mp.audio_path, figure=plt.figure(),
                        sound=self.sound, size=mp.size, size_hint=(None, None)
                    )
                    for mp in sct_miniplots
                ]

                _ = [child.reset() for child in self.ids.box_signals.children]
                self.ids.box_signals.clear_widgets()

                for i, mp in enumerate(sct_miniplots):
                    i_widget = MDIcon(
                        icon=f'numeric-{i}-box',
                        theme_text_color='Custom', text_color=colors['Blue']['A400']
                    )
                    i_widget.pos_hint = {'x': .05, 'y': .1}
                    i_widget.size_hint = (.25, None)
                    mp.add_widget(i_widget)
                    self.ids.box_signals.add_widget(mp)

                check_synchronization_args = self.parent_tab.ids.select.get_check_synchronization_args()

                peaks, results = {}, {}
                for comb in combs:
                    A_idx, B_idx = comb
                    At = self.ids.box_signals.children[-A_idx-1].audio_data
                    Bt = self.ids.box_signals.children[-B_idx-1].audio_data

                    result = check_synchronization(At, Bt, ana_fs, **check_synchronization_args)
                    peaks_dict = result.pop('peaks')

                    results[str(comb)] = result

                    if A_idx not in peaks:
                        peaks[A_idx] = peaks_dict['A']
                    if B_idx not in peaks:
                        peaks[B_idx] = peaks_dict['B']

                self.peaks, self.results = peaks, results

                result_treeview = self.ids.result_treeview

                _ = [
                    result_treeview.remove_node(node)
                    for node in list(result_treeview.iterate_all_nodes())
                ]

                for label, result in sorted(results.items(), key=lambda x: x[1]['n'], reverse=True):
                    result_node = result_treeview.add_node(TreeViewLabel(text=label))
                    _ = [
                        result_treeview.add_node(
                            TreeViewLabel(text=f'{k}: {result[k]}'), parent=result_node
                        )
                        for k in ['n', 'V']
                    ]

                def on_selected_node(instance, value):
                    if value.text in self.results:
                        result = self.results[value.text]

                        fig_hist = self.fig_hist
                        fig_hist.clear()
                        ax_hist = fig_hist.add_subplot()

                        ax_hist.hist(result['phis'], bins=8, range=(0, 2*torch.pi))
                        ax_hist.set_xticks(torch.arange(0, 2*torch.pi+1e-6, torch.pi/2))
                        ax_hist.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

                        ax_hist.patch.set_alpha(0)

                        fig_hist.canvas.draw()

                        indices = eval(value.text)
                        alpha_list = [
                            .5 if idx in indices else .0
                            for idx in range(len(self.ids.box_signals.children))
                        ][::-1]
                        _ = [
                            [mp.figure.axes[0].patch.set_alpha(alpha), mp.figure.canvas.draw()]
                            for alpha, mp in zip(alpha_list, self.ids.box_signals.children)
                        ]

                if self.treeview_callback is not None:
                    result_treeview.unbind(selected_node=self.treeview_callback)

                self.treeview_callback = on_selected_node
                self.ids.result_treeview.bind(selected_node=self.treeview_callback)

                for idx, peaks_val in peaks.items():
                    audio_miniplot = self.ids.box_signals.children[-idx-1]
                    audio_data, audio_fs = audio_miniplot.audio_data, ana_fs

                    fig_wave = audio_miniplot.figure
                    ax_wave = fig_wave.add_subplot(facecolor='r')

                    show_wave(audio_data, audio_fs, ax=ax_wave, color='b')
                    ax_wave.plot(
                        peaks_val/audio_fs, audio_data[peaks_val], marker='x', color='yellow', linewidth=0
                    )

                    ax_wave.set_xlim(0, audio_data.size(-1)/audio_fs)
                    ax_wave.tick_params(
                        which='both', axis='both',
                        top=True, labeltop=True, bottom=False, labelbottom=False,
                        left=False, labelleft=False, right=False, labelright=False,
                    )
                    ax_wave.set_xlabel(''); ax_wave.set_ylabel('')
                    ax_wave.patch.set_alpha(0)

                    fig_wave.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, wspace=0, hspace=0)
                    fig_wave.patch.set_alpha(0)

                    fig_wave.canvas.draw()

                _ = [
                    result_treeview.select_node(node)
                    for i, node in enumerate(self.ids.result_treeview.iterate_all_nodes())
                    if i == 1
                ]

    def save(self):
        if self.audio_dict and self.results and self.peaks:
            selections = filechooser.choose_dir(use_extensions=True)

            if selections:
                path = pathlib.Path(selections[0])
                audio_fs = self.audio_dict['fs']
                audio_label = self.audio_dict['label']

                peak_csv = [['frog_index', 'peak_time']]
                _ = [
                    [peak_csv.append([k, t]) for t in (v/audio_fs).tolist()]
                    for k, v in self.peaks.items()
                ]
                with open(str(path/f'{audio_label}_peak.csv'), 'w') as peak_f:
                    writer = csv.writer(peak_f, delimiter='\t')
                    writer.writerows(peak_csv)

                phi_csv = [['combination', 'phi']]
                _ = [
                    [phi_csv.append([k, p]) for p in v['phis'].tolist()]
                    for k, v in self.results.items()
                ]
                with open(str(path/f'{audio_label}_phi.csv'), 'w') as phi_f:
                    writer = csv.writer(phi_f, delimiter='\t')
                    writer.writerows(phi_csv)

class FrogAudioDisplay(MDBoxLayout):
    audio_dict = DictProperty({})

    def on_audio_dict(self, instance, value):
        audio_timeline, audio_toolbar = self.ids.audio_timeline, self.ids.audio_toolbar
        audio_timeline.audio_dict = audio_toolbar.audio_dict = value