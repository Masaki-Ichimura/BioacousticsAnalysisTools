import torch

from kivy.lang import Builder

from utils.audio import silence_pydub, silence_pyaudioanalysis
from utils.audio.transform import apply_freq_mask
from app.gui.widgets.tab import Tab

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/silence_removal.kv')

class SilenceRemovalTab(Tab):
    prob_dict = None

    def on_kv_post(self, *arg, **kwargs):
        self.ids.screen_manager.current = 'svm_thr'

    def get_audio(self):
        audio_detail = self.parent.parent.parent.parent.parent.parent
        working_container = audio_detail.parent.parent

        if not working_container.audio_file:
            return None, None
        else:
            return working_container.audio_data, working_container.audio_fs

    def get_freq_args(self):
        if self.ids.limit_freq.active:
            freq_high = self.ids.freq_high.text
            freq_low = self.ids.freq_low.text
        else:
            freq_high = None
            freq_low = None

        freq_args = dict(
            freq_high=int(freq_high) if freq_high else None,
            freq_low=int(freq_low) if freq_low else None
        )
        return freq_args

    def get_mode(self):
        if self.ids.svm_checkbox.active:
            return 'svm'
        elif self.ids.rms_checkbox.active:
            return 'rms'

    def get_func(self):
        mode = self.get_mode()

        if mode == 'svm':
            return silence_pyaudioanalysis.silence_removal
        elif mode == 'rms':
            return silence_pydub.silence_removal

    def get_func_args(self):
        mode = self.get_mode()

        if mode == 'svm':
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
        elif mode == 'rms':
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

    def plot(
        self, signal, sample_rate, nonsilent_sections, ax,
        probability=None, threshold=None
    ):
        d_min, d_max = signal.mean(0).min().item(), signal.mean(0).max().item()
        d_min, d_max = min(d_min, -abs(d_max)), max(d_max, abs(d_min))

        nonsilence = d_min*torch.ones(signal.shape[-1]*1000//sample_rate)
        for sec in nonsilent_sections:
            nonsilence[sec[0]:sec[1]] = d_max

        x_ms = torch.linspace(0, signal.size(-1)/sample_rate, steps=nonsilence.size(0))
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
        audio_data, audio_fs = self.get_audio()

        if audio_data is None:
            return None

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
        self.prob_dict = prob_dict
        print(nonsilent_sections)

        ax_wave = audio_timeline.fig_wave.axes[0]

        self.clear()
        self.plot(xnt, audio_fs, nonsilent_sections, ax_wave, **prob_dict)
        audio_timeline.fig_wave.canvas.draw()

    def replot_button_clicked(self):
        mode = self.get_mode()
        audio_data, audio_fs = self.get_audio()
        prob_dict = self.prob_dict

        audio_detail = self.parent.parent.parent.parent.parent.parent
        working_container = audio_detail.parent.parent
        audio_timeline = working_container.ids.audio_display.ids.audio_timeline

        if mode == 'svm':
            if prob_dict and audio_data is not None:
                func_args = self.get_func_args()
                nonsilent_sections = silence_pyaudioanalysis.segmentation(
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
                self.plot(
                    audio_data, audio_fs, nonsilent_sections, ax_wave,
                    **prob_dict
                )
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
