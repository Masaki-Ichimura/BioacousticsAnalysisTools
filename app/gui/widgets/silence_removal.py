import torch

from kivy.lang import Builder
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.tab import MDTabsBase

from utils.audio import silence_pydub, silence_pyaudioanalysis
from utils.audio.transform import freq_mask

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/silence_removal.kv')


class Tab(MDFloatLayout, MDTabsBase):
    pass


class SilenceRemovalTab(Tab):
    def get_freq_args(self):
        args = dict(
            freq_high=int(self.ids.freq_high.text),
            freq_low=int(self.ids.freq_low.text)
        )
        return args

    def plot_nonsilent_sections(
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
            rmsilence_args = self.get_rmsilence_args()
            pr_ms = probability[:, None].tile((1, rmsilence_args['seek_ms'])).view(-1)
            pr_ms = torch.nn.functional.pad(
                pr_ms[:x_ms.size(0)],
                [0, x_ms.size(0)-min(pr_ms.size(0), x_ms.size(0))],
                'constant', torch.nan
            )

            ax.plot(x_ms, d_min+(d_max-d_min)*pr_ms, color='yellowgreen')

        if threshold is not None:
            ax.axhline(y=d_min+(d_max-d_min)*threshold, color='yellow')


    def ok_button_clicked(self):
        audio_detail = self.parent.parent.parent.parent.parent.parent
        working_container = audio_detail.parent.parent
        audio_timeline = working_container.ids.audio_display.ids.audio_timeline

        main_menu = self.get_root_window().children[0].ids.main_menu
        config_container = main_menu.ids.config_container

        audio_data = working_container.audio_data
        audio_fs = working_container.audio_fs

        if not working_container.audio_file:
            return None

        stft_args = config_container.get_stft_args()
        freq_args = self.get_freq_args()
        rmsilence_args = self.get_rmsilence_args()

        xnt = audio_data

        if self.ids.limit_freq.active:
            Xnkl = torch.stft(xnt, return_complex=True, **stft_args)
            mk = freq_mask(
                audio_fs, stft_args['n_fft'],
                freq_low=freq_args['freq_low'], freq_high=freq_args['freq_high']
            )
            Xnkl[:, mk] = 0
            ynt = torch.istft(Xnkl, length=xnt.size(-1), **stft_args)
        else:
            ynt = xnt

        nonsilent_sections, prob_dict = self.rmsilence_func(ynt, audio_fs, **rmsilence_args)
        print(nonsilent_sections)

        ax_wave = audio_timeline.fig_wave.axes[0]

        self.reset_button_clicked()

        self.plot_nonsilent_sections(ynt, audio_fs, nonsilent_sections, ax_wave, **prob_dict)

        audio_timeline.fig_wave.canvas.draw()

    def reset_button_clicked(self):
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


class PydubBasedSilenceRemovalTab(SilenceRemovalTab):
    def on_kv_post(self, *arg, **kwargs):
        self.rmsilence_func = silence_pydub.silence_removal

    def get_rmsilence_args(self):
        args = dict(
            min_silence_ms=int(self.ids.min_silence_ms.text),
            seek_ms=int(self.ids.seek_ms.text),
            threshold=float(self.ids.threshold.text),
            return_prob=True
        )
        return args


class PyAudioAnalysisBasedSilenceRemovalTab(SilenceRemovalTab):
    def on_kv_post(self, *arg, **kwargs):
        self.rmsilence_func = silence_pyaudioanalysis.silence_removal

    def get_rmsilence_args(self):
        args = dict(
            win_ms=int(self.ids.win_ms.text),
            seek_ms=int(self.ids.seek_ms.text),
            freq_low=int(self.ids.freq_low.text),
            freq_high=int(self.ids.freq_high.text),
            smooth_window_ms=int(self.ids.smooth_window_ms.text),
            min_nonsilence_ms=int(self.ids.min_nonsilence_ms.text),
            weight=float(self.ids.weight.text),
            return_prob=True
        )
        if not self.ids.limit_freq.active:
            _ = [args.pop(k) for k in ['freq_low', 'freq_high']]
        return args
