import torch

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import DictProperty, ListProperty

from batools.utils.audio.silence_removal import silence_removal, segmentation
from batools.utils.audio.transform import apply_freq_mask, extract_from_section
from batools.app.gui.widgets.sub_tab import SubTab

Builder.load_file(__file__[:-3]+'.kv')


class SilenceRemovalTab(SubTab):
    audio_dict = DictProperty({})
    prob_dict = None
    nonsilent_sections = ListProperty([])

    def on_audio_dict(self, instance, value):
        option_names = [
            'freqfilter', 'minimum_nonsilence', 'broadened_nonsilence',
            'window', 'hop', 'weight', 'smooth_window'
        ]
        option_values = {
            'freqfilter_min': f'{"" if not value else 0}',
            'freqfilter_max': f'{"" if not value else value["fs"]//2}',
            'minimum_nonsilence': f'{200}',
            'broadened_nonsilence': f'{0}',
            'window': f'{1000}',
            'hop': f'{500}',
            'smooth_window': f'{500}',
            'weight': f'{0.5}'
        }

        _ = [
            setattr(getattr(self.ids, f'{name}_checkbox'), 'disabled', not value)
            for name in option_names
        ]
        _ = [
            setattr(getattr(self.ids, f'{name}_value'), 'text', opt_value)
            for name, opt_value in option_values.items()
        ]
        self.nonsilent_sections = []
        self.prob_dict = None

    def on_nonsilent_sections(self, instance, value):
        app = App.get_running_app()
        working_container = app.links['edit_tab'].ids.working_container
        preprocessed = working_container.ids.audio_detail.ids.preprocessed

        preprocessed.ids.silence_removal_checkbox.disabled = not value

    def get_func_args(self):
        func_args = dict(
            win_ms=1000,
            seek_ms=500,
            weight=.5,
            smooth_window_ms=500,
            broaden_section_ms=0,
            min_nonsilence_ms=200,
            freq_high=None,
            freq_low=None,
            return_prob=True
        )

        if self.ids.freqfilter_checkbox.state == 'down':
            try:
                audio_fs = self.audio_dict['fs']
                freq_high = max(min(int(self.ids.freqfilter_max_value.text), audio_fs//2), 0)
                freq_low = min(max(int(self.ids.freqfilter_min_value.text), 0), audio_fs//2)
            except ValueError:
                freq_high, freq_low = None, None

            if freq_high is not None or freq_low is not None:
                func_args.update(dict(freq_high=freq_high, freq_low=freq_low))

        if self.ids.minimum_nonsilence_checkbox.state == 'down':
            try:
                min_nonsilence_ms = max(int(self.ids.minimum_nonsilence_value.text), 0)
            except ValueError:
                min_nonsilence_ms = None

            if min_nonsilence_ms is not None:
                func_args.update(dict(min_nonsilence_ms=min_nonsilence_ms))

        if self.ids.broadened_nonsilence_checkbox.state == 'down':
            try:
                broaden_section_ms = max(int(self.ids.broadened_nonsilence_value.text), 0)
            except ValueError:
                broaden_section_ms = None

            if broaden_section_ms is not None:
                func_args.update(dict(broaden_section_ms=broaden_section_ms))

        if self.ids.window_checkbox.state == 'down':
            try:
                win_ms = max(int(self.ids.window_value.text), 1)
            except ValueError:
                win_ms = None

            if win_ms is not None:
                func_args.update(dict(win_ms=win_ms))

        if self.ids.hop_checkbox.state == 'down':
            try:
                seek_ms = max(int(self.ids.hop_value.text), 1)
            except ValueError:
                seek_ms = None

            if seek_ms is not None:
                func_args.update(dict(seek_ms=seek_ms))

        if self.ids.weight_checkbox.state == 'down':
            try:
                weight = min(max(int(self.ids.weight_value.text), .01), .99)
            except ValueError:
                weight = None

            if weight is not None:
                func_args.update(dict(weight=weight))

        if self.ids.smooth_window_checkbox.state == 'down':
            try:
                smooth_window_ms = max(int(self.ids.smooth_window_value), func_args['seek_ms'])
            except ValueError:
                smooth_window_ms = None

            if smooth_window_ms is not None:
                func_args.update(dict(smooth_window_ms=smooth_window_ms))

        return func_args

    def clear(self):
        audio_detail = self.parent_tab
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
            audio_detail = self.parent_tab
            working_container = audio_detail.parent.parent
            audio_timeline = working_container.ids.audio_display.ids.audio_timeline

            func_args = self.get_func_args()

            xnt = audio_data

            if self.ids.freqfilter_checkbox.state == 'down':
                ynt = apply_freq_mask(
                    xnt, audio_fs, freq_high=func_args['freq_high'], freq_low=func_args['freq_low']
                )
            else:
                ynt = xnt

            nonsilent_sections, prob_dict = silence_removal(ynt, audio_fs, **func_args)
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

        if audio_data is not None and prob_dict:
            func_args = self.get_func_args()
            self.nonsilent_sections = segmentation(
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
            prob_dict['threshold'] = prob_dict['threshold'] + eval(val)

        self.replot_button_clicked()

    def extract(self, label):
        nonsilent_sections = self.nonsilent_sections
        audio_data, audio_fs = self.audio_dict['data'], self.audio_dict['fs']

        if audio_data is not None and nonsilent_sections:
            app = App.get_running_app()
            cache_dir = app.tmp_dir
            extracted_dicts = []
            for section in nonsilent_sections:
                audio_label = f'{label}_{section[0]}-{section[1]}'
                audio_cache = f'{cache_dir.name}/tmp_{audio_label}.wav'

                extracted_dicts.append(dict(
                    label=audio_label, path='', cache=audio_cache,
                    data=extract_from_section(audio_data, audio_fs, section),
                    fs=audio_fs, ch=-1
                ))

            return extracted_dicts