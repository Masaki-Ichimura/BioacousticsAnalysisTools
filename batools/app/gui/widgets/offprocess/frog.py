import torch
import torchaudio
import threading
import matplotlib.pyplot as plt
from itertools import combinations

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import DictProperty
from kivy.uix.treeview import TreeViewLabel
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivymd.color_definitions import colors
from kivymd.uix.screen import MDScreen
from kivymd.uix.label.label import MDIcon
from kivymd.uix.selectioncontrol.selectioncontrol import MDCheckbox
from kivymd.uix.segmentedcontrol.segmentedcontrol import MDSegmentedControlItem

from batools.app.gui.widgets.sub_tab import SubTab
from batools.app.gui.widgets.audiodisplay import AudioMiniplot
from batools.utils.audio.analysis.frog import check_synchronization
from batools.utils.audio.bss.auxiva import AuxIVA
from batools.utils.audio.bss.fastmnmf import FastMNMF
from batools.utils.audio.bss.ilrma import ILRMA
from batools.utils.audio.plot import show_wave

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

        # これをやってもスイッチの位置が更新されないが，一応残しておく
        default_segment = [
            child for child in self.ids.mode_control.ids.segment_panel.children
            if type(child) is MDSegmentedControlItem and child.text.lower() == self.mode
        ][0]
        self.ids.mode_control.on_press_segment(default_segment, default_segment)

        # init コードブロック内のクロックにより width の値が逐次更新されるため，こちらからスイッチの幅を変えるのは不可能
        # kivyMD側の対応を待つしかない (特に報告はしてないが，コードを見る限り既にバグとして認知されてそうな雰囲気)
        # 最悪，kivymd.uix.segmentedcontrol.MDSegmentedControl の init を書き換えれば何とかなりそうだが
        # self.ids.mode_control.ids.segment_switch.width = self.ids.mode_control.ids.segment_panel.children[0].width

    def on_audio_dict(self, instance, value):
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
                self.parent_tab.ids.select.audio_dict = self.sep_dict
                self.parent_tab.ids.screen_manager.current = 'select'
                self.ids.separate_button.disabled = False
                self.ids.mode_control.disabled = False
                self.ids.progressbar.value = 0

            thread = threading.Thread(target=separate_process)
            thread.start()

class FrogSelect(MDScreen):
    audio_dict = DictProperty({})
    checkboxes = []

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
                torchaudio.save(filepath=ch_path.format(ch), src=ch_data[None], sample_rate=sep_fs)

                audio_miniplot = AudioMiniplot(
                    data=ch_data, fs=sep_fs, path=ch_path.format(ch), size_hint=(1/3, 1/3)
                )

                checkbox_widget = MDCheckbox()
                checkbox_widget.selected_color = checkbox_widget.unselected_color = colors['Blue']['A400']
                checkbox_widget.pos_hint = {'x': .0, 'top': .4}
                checkbox_widget.size_hint = (.25, None)

                audio_miniplot.add_widget(checkbox_widget)
                checkboxes.append(checkbox_widget)

                self.ids.stack_sep.add_widget(audio_miniplot)

            self.checkboxes = checkboxes

    def select(self):
        indices = [checkbox.active for checkbox in self.checkboxes]
        if self.audio_dict and any(indices):
            app = App.get_running_app()
            cache_dir = app.tmp_dir

            sct_label = f'select_{self.audio_dict["label"]}'
            sct_data, sct_fs = self.audio_dict['data'][indices], self.audio_dict['fs']
            sct_cache = f'{cache_dir.name}/{sct_label}.wav'

            sct_dict = dict(
                label=sct_label, path=None, cache=sct_cache, data=sct_data, fs=sct_fs, ch=-1
            )

            self.parent_tab.ids.analysis.audio_dict = sct_dict

class FrogAnalysis(MDScreen):
    audio_dict = DictProperty({})

    def on_kv_post(self, *args, **kwargs):
        self.fig_hist = plt.figure()
        self.treeview_callback = None

        self.fig_hist.patch.set_alpha(0)
        self.ids.box_hist.add_widget(FigureCanvasKivyAgg(self.fig_hist))

    def on_audio_dict(self, instance, value):
        if self.audio_dict:
            ana_datas, ana_fs = self.audio_dict['data'], self.audio_dict['fs']

            if ana_datas.size(0) > 1:
                combs = combinations(range(ana_datas.size(0)), 2)

                sep_miniplots = self.parent_tab.ids.select.ids.stack_sep.children
                sep_datas = [mp.audio_data for mp in sep_miniplots]

                sct_miniplots = [
                    sep_miniplots[[ana_data.equal(sep_data) for sep_data in sep_datas].index(True)]
                    for ana_data in ana_datas
                ]

                sct_miniplots = [
                    AudioMiniplot(
                        data=mp.audio_data, fs=mp.audio_fs, path=mp.audio_path, figure=plt.figure(),
                        size=mp.size, size_hint=(None, None)
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

                peaks_tmp, results = {}, {}
                for comb in combs:
                    A_idx, B_idx = comb
                    At = self.ids.box_signals.children[-A_idx-1].audio_data
                    Bt = self.ids.box_signals.children[-B_idx-1].audio_data

                    result = check_synchronization(At, Bt, ana_fs)
                    peaks_dict = result.pop('peaks')

                    results[str(comb)] = result

                    if A_idx not in peaks_tmp:
                        peaks_tmp[A_idx] = peaks_dict['A']
                    if B_idx not in peaks_tmp:
                        peaks_tmp[B_idx] = peaks_dict['B']

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
                    if value.text in results:
                        result = results[value.text]

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

                for idx, peaks in peaks_tmp.items():
                    audio_miniplot = self.ids.box_signals.children[-idx-1]
                    audio_data, audio_fs = audio_miniplot.audio_data, ana_fs

                    fig_wave = audio_miniplot.figure
                    ax_wave = fig_wave.add_subplot(facecolor='r')

                    show_wave(audio_data, audio_fs, ax=ax_wave, color='b')
                    ax_wave.plot(
                        peaks/audio_fs, audio_data[peaks], marker='x', color='yellow', linewidth=0
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
