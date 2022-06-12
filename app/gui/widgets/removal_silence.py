import torch

from kivy.lang import Builder
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.tab import MDTabsBase

from utils.audio import silence_pydub, silence_pyaudioanalysis

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/removal_silence.kv')


class Tab(MDFloatLayout, MDTabsBase):
    pass


class RemovalSilenceTab(Tab):
    def get_freq_args(self):
        args = dict(
            freq_high=int(self.ids.freq_high.text),
            freq_low=int(self.ids.freq_low.text)
        )
        return args

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
            k = torch.fft.rfftfreq(stft_args['n_fft']) * audio_fs
            hk = torch.logical_or(k<freq_args['freq_low'], k>freq_args['freq_high'])
            Xnkl[:, hk] = 0
            ynt = torch.istft(Xnkl, length=audio_data.size(-1), **stft_args)
        else:
            ynt = xnt

        nonsilent_sections = self.rmsilence_func(ynt, audio_fs, **rmsilence_args)

        print(nonsilent_sections)

        d_min, d_max = xnt.mean(0).min().item(), xnt.mean(0).max().item()
        d_min, d_max = min(d_min, -abs(d_max)), max(d_max, abs(d_min))
        nonsilence = d_min*torch.ones(audio_data.size(-1)*1000//audio_fs)
        for sec in nonsilent_sections:
            nonsilence[sec[0]:sec[1]] = d_max

        ax_wave = audio_timeline.fig_wave.axes[0]

        if len(ax_wave.collections) > 1:
            ax_wave.collections[-1].remove()

        ax_wave.fill_between(
            torch.linspace(0, audio_data.size(-1)/audio_fs, steps=nonsilence.size(0)),
            nonsilence, d_min,
            facecolor='r', alpha=.5
        )
        audio_timeline.fig_wave.canvas.draw()

    def reset_button_clicked(self):
        audio_detail = self.parent.parent.parent.parent.parent.parent
        working_container = audio_detail.parent.parent
        audio_timeline = working_container.ids.audio_display.ids.audio_timeline
        fig_wave = audio_timeline.fig_wave

        if not fig_wave:
            return None

        ax_wave = fig_wave.axes[0]

        if len(ax_wave.collections)>1:
            ax_wave.collections[-1].remove()

        fig_wave.canvas.draw()


class PydubBasedRemovalSilenceTab(RemovalSilenceTab):
    def on_kv_post(self, *arg, **kwargs):
        self.rmsilence_func = silence_pydub.detect_nonsilent

    def get_rmsilence_args(self):
        args = dict(
            min_silence_len=int(self.ids.min_silence_len.text),
            silence_thresh=float(self.ids.silence_thresh.text),
            seek_step=int(self.ids.seek_step.text)
        )
        return args


class PyAudioAnalysisBasedRemovalSilenceTab(RemovalSilenceTab):
    def on_kv_post(self, *arg, **kwargs):
        self.rmsilence_func = silence_pyaudioanalysis.silence_removal

    def get_rmsilence_args(self):
        args = dict(
            win_msec=int(self.ids.win_msec.text),
            seek_msec=int(self.ids.seek_msec.text),
            freq_low=int(self.ids.freq_low.text),
            freq_high=int(self.ids.freq_high.text),
            smooth_window_msec=int(self.ids.smooth_window_msec.text),
            min_duration_msec=int(self.ids.min_duration_msec.text),
            weight=float(self.ids.weight.text)
        )
        if not self.ids.limit_freq.active:
            _ = [args.pop(k) for k in ['freq_low', 'freq_high']]
        return args
