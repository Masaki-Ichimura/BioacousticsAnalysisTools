import torch

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import DictProperty, ListProperty

from batools.utils.audio import silence_pydub, silence_pyaudioanalysis
from batools.utils.audio.transform import apply_freq_mask, extract_from_section
from batools.app.kivy_utils import TorchTensorProperty
from batools.app.gui.widgets.sub_tab import SubTab

Builder.load_file(__file__[:-3]+'.kv')


class SilenceRemovalTab(SubTab):
    audio_dict = DictProperty({})
    prob_dict = None
    nonsilent_sections = ListProperty([])
    mode = 'svm'

    def on_kv_post(self, *args, **kwargs):
        self.ids.screen_manager.current = 'svm_thr'

    def on_audio_dict(self, instance, value):
        if value:
            label = value['label']
            if '.' in label:
                label = label[:-label[::-1].index('.')-1]
        else:
            label = ''

        self.ids.label.text = label
        self.nonsilent_sections = []
        self.prob_dict = None

    def on_nonsilent_sections(self, instance, value):
        app = App.get_running_app()
        working_container = app.links['edit_tab'].ids.working_container
        preprocessed = working_container.ids.audio_detail.ids.preprocessed

        preprocessed.ids.silence_removal_checkbox.disabled = not value

    def get_freq_args(self):
        if self.ids.limit_freq.active:
            freq_high, freq_low = int(self.ids.freq_high.text), int(self.ids.freq_low.text)
        else:
            freq_high, freq_low = None, None

        return dict(freq_high=freq_high, freq_low=freq_low)

    def get_func(self):
        if self.mode == 'svm':
            return silence_pyaudioanalysis.silence_removal
        elif self.mode == 'rms':
            return silence_pydub.silence_removal

    def get_func_args(self):
        if self.mode == 'svm':
            func_args = dict(
                win_ms=int(self.ids.svm_win.text),
                seek_ms=int(self.ids.svm_seek.text),
                weight=float(self.ids.svm_weight.text),
                smooth_window_ms=int(self.ids.svm_smooth_window.text),
                broaden_section_ms=int(self.ids.svm_broaden.text),
                min_nonsilence_ms=int(self.ids.svm_min_nonsilence.text),
                return_prob=True
            )
            func_args.update(self.get_freq_args())
        elif self.mode == 'rms':
            func_args = dict(
                min_silence_ms=int(self.ids.rms_min_silence.text),
                seek_ms=int(self.ids.rms_seek.text),
                threshold=float(self.ids.rms_threshold.text),
                return_prob=True
            )

        return func_args

    def clear(self):
        audio_detail = self.parent.parent.parent.parent.parent.parent
        working_container = audio_detail.parent.parent
        audio_timeline = working_container.ids.audio_display.ids.audio_timeline
        fig_wave = audio_timeline.fig_wave

        if not fig_wave:
            return None

        ax_wave = fig_wave.axes[0]

        if len(ax_wave.collections) > 1:
            _ = [collection.remove() for collection in ax_wave.collections[1:]]

        if len(ax_wave.lines) > 1:
            _ = [line.remove() for line in ax_wave.lines[1:]]

        fig_wave.canvas.draw()

    def plot(self, ax, probability=None, threshold=None):
        audio_data, audio_fs = self.audio_dict['data'], self.audio_dict['fs']
        nonsilent_sections = self.nonsilent_sections

        d_min, d_max = audio_data.mean(0).min().item(), audio_data.mean(0).max().item()
        d_min, d_max = min(d_min, -abs(d_max)), max(d_max, abs(d_min))

        nonsilence = d_min*torch.ones(audio_data.size(-1)*1000//audio_fs)
        for section in nonsilent_sections:
            nonsilence[section[0]:section[1]] = d_max

        x_ms = torch.linspace(0, audio_data.size(-1)/audio_fs, steps=nonsilence.size(0))
        ax.fill_between(x_ms, nonsilence, d_min, facecolor='r', alpha=.5)

        if probability is not None:
            rmsilence_args = self.get_func_args()
            pr_ms = probability[:, None].tile((1, rmsilence_args['seek_ms'])).view(-1)
            pr_ms = torch.nn.functional.pad(
                pr_ms[:x_ms.size(0)],
                [0, x_ms.size(0)-min(pr_ms.size(0), x_ms.size(0))],
                'constant', torch.nan
            )

            ax.plot(x_ms, d_min+(d_max-d_min)*pr_ms, linewidth=2., color='yellowgreen')

        if threshold is not None:
            ax.axhline(y=d_min+(d_max-d_min)*threshold, linewidth=2., color='yellow')

    def plot_button_clicked(self):
        audio_data, audio_fs = self.audio_dict['data'], self.audio_dict['fs']

        if audio_data is not None:
            audio_detail = self.parent.parent.parent.parent.parent.parent
            working_container = audio_detail.parent.parent
            audio_timeline = working_container.ids.audio_display.ids.audio_timeline

            func = self.get_func()
            func_args = self.get_func_args()
            freq_args = self.get_freq_args()

            xnt = audio_data

            if self.ids.limit_freq.active:
                ynt = apply_freq_mask(xnt, audio_fs, **freq_args)
            else:
                ynt = xnt

            nonsilent_sections, prob_dict = func(ynt, audio_fs, **func_args)
            self.nonsilent_sections, self.prob_dict = nonsilent_sections, prob_dict

            ax_wave = audio_timeline.fig_wave.axes[0]

            self.clear()
            self.plot(ax_wave, **prob_dict)
            audio_timeline.fig_wave.canvas.draw()

    def replot_button_clicked(self):
        audio_data, audio_fs = self.audio_dict['data'], self.audio_dict['fs']
        prob_dict = self.prob_dict

        audio_detail = self.parent.parent.parent.parent.parent.parent
        working_container = audio_detail.parent.parent
        audio_timeline = working_container.ids.audio_display.ids.audio_timeline

        if self.mode == 'svm':
            if audio_data is not None and prob_dict:
                func_args = self.get_func_args()
                self.nonsilent_sections = silence_pyaudioanalysis.segmentation(
                    probability=prob_dict['probability'],
                    threshold=prob_dict['threshold'],
                    seek_num=func_args['seek_ms'],
                    clustering=True,
                    broaden_section_num=func_args['broaden_section_ms'],
                    enable_merge=True,
                    min_duration_num=func_args['min_nonsilence_ms']
                )
                ax_wave = audio_timeline.fig_wave.axes[0]

                self.clear()
                self.plot(ax_wave, **prob_dict)
                audio_timeline.fig_wave.canvas.draw()

    def change_threshold_button_clicked(self, val: str):
        prob_dict = self.prob_dict

        if prob_dict:
            if val[0] == '+':
                prob_dict['threshold'] += float(val[1:])
            elif val[0] == '-':
                prob_dict['threshold'] -= float(val[1:])
            else:
                prob_dict['threshold'] = float(val)

        self.replot_button_clicked()

    def extract(self):
        nonsilent_sections = self.nonsilent_sections
        audio_data, audio_fs = self.audio_dict['data'], self.audio_dict['fs']

        if audio_data is not None and nonsilent_sections:
            app = App.get_running_app()
            cache_dir = app.tmp_dir
            extracted_dicts = []
            for section in nonsilent_sections:
                audio_label = f'{self.ids.label.text}_{section[0]}-{section[1]}'
                audio_cache = f'{cache_dir.name}/tmp_{audio_label}.wav'

                extracted_dicts.append(dict(
                    label=audio_label, path='', cache=audio_cache,
                    data=extract_from_section(audio_data, audio_fs, section),
                    fs=audio_fs, ch=-1
                ))

            return extracted_dicts

        else: []

            # audio_detail = self.parent.parent.parent.parent.parent.parent
            # preprocessed = audio_detail.ids.preprocessed

            # preprocessed.audio_dicts.extend(extracted_dicts)

            # tabs = audio_detail.ids.tabs
            # tabs.switch_tab('format-list-bulleted', search_by='icon')
