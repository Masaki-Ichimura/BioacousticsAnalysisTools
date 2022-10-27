import torch
import torchaudio
import gc
import matplotlib.pyplot as plt

from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import DictProperty, NumericProperty, ObjectProperty
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.core.audio.audio_ffpyplayer import SoundFFPy
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.progressbar.progressbar import MDProgressBar
from kivymd.uix.button.button import MDFloatingActionButton

from batools.app.gui.widgets.container import Container
from batools.app.kivy_utils import TorchTensorProperty
from batools.utils.audio.plot import show_spec, show_wave

Builder.load_file(__file__[:-3]+'.kv')


class AudioTimeline(Container):
    audio_dict = DictProperty({})
    audio_data_org = TorchTensorProperty(torch.zeros(1))
    audio_data = TorchTensorProperty(torch.zeros(1))
    audio_fs = None
    audio_toolbar = None

    fig_wave = ObjectProperty(None)
    fig_spec = ObjectProperty(None)

    timeline_t_unit = NumericProperty(1.)
    timeline_width = NumericProperty(1000)

    sound = None
    audio_pos = NumericProperty(0)

    check_dt = .05

    label_size = 8
    majortick_length = 5
    minortick_length = majortick_length / 2

    def on_kv_post(self, *args, **kwargs):

        def touch_up_timeline(instance, event):
            if self.audio_dict and event.button == 'left':
                sound_length = self.sound.length
                fig_width = instance.width

                self.audio_pos = max(0, min(event.pos[0], fig_width)/fig_width*sound_length)

        self.ids.box_tl.bind(on_touch_up=touch_up_timeline)

        fig_wave, ax_wave = plt.subplots()
        fig_spec, ax_spec = plt.subplots()
        self.fig_wave, self.fig_spec = fig_wave, fig_spec
        self.fig_t, self.fig_y, self.fig_f = [plt.figure() for _ in range(3)]

    def on_audio_dict(self, instance, value):
        if value:
            audio_dict = self.audio_dict
            self.audio_data_org = audio_dict['data']

            audio_ch = audio_dict['ch']
            self.audio_fs = audio_dict['fs']
            if audio_ch >= 0:
                self.audio_data = audio_dict['data'][audio_ch, None]
            else:
                self.audio_data = audio_dict['data'].mean(0, keepdim=True)
        else:
            # init timeline
            seekbar = self.ids.seekbar
            self.audio_pos = 0
            if self.sound:
                self.sound.unload()
            self.ids.box_tl.clear_widgets()
            self.ids.box_yaxis.clear_widgets()
            self.ids.box_tl.add_widget(seekbar)

    def on_audio_data_org(self, instance, value):
        audio_dict = self.audio_dict
        if audio_dict:
            chs = list(range(-1, value.size(0))) if value.size(0) > 1 else [0]

            if self.audio_toolbar:
                items = [
                    dict(
                        viewclass='OneLineListItem',
                        text=f'{ch:02d}ch' if ch >= 0 else 'mean',
                        height=dp(54),
                        on_release=lambda x=ch: self.audio_toolbar.set_ch(x)
                    )
                    for ch in chs
                ]
                self.audio_toolbar.ch_window_menu = MDDropdownMenu(
                    caller=self.audio_toolbar.ids.ch, width_mult=4, items=items
                )
                def on_text(instance, value):
                    if audio_dict:
                        if value[:2].isdigit():
                            audio_dict['ch'] = int(value[:2])
                        else:
                            audio_dict['ch'] = -1

                self.audio_toolbar.ids.ch.bind(text=on_text)
                if audio_dict['ch'] == -1:
                    self.audio_toolbar.ids.ch.text = items[0]['text']
                else:
                    self.audio_toolbar.ids.ch.text = f'{audio_dict["ch"]:02d}ch'

    def on_audio_data(self, instance, value):
        audio_data, audio_fs = self.audio_data, self.audio_fs
        audio_path, audio_cache = self.audio_dict['path'], self.audio_dict['cache']
        if not audio_path:
            audio_path = audio_cache
            torchaudio.save(filepath=audio_path, src=audio_data, sample_rate=audio_fs)

        self.init_timeline()

        t_unit = self.timeline_t_unit
        t_start, t_end = 0, int(audio_data.size(-1)/(audio_fs*t_unit))*t_unit
        t_ticks = torch.arange(t_start, t_end+t_unit*.1, t_unit)

        fig_wave, ax_wave = self.fig_wave, self.fig_wave.add_subplot()
        show_wave(audio_data, audio_fs, ax=ax_wave, color='b')
        ax_wave.set_xticks(t_ticks)
        ax_wave.set_xlim(0, audio_data.shape[-1]/audio_fs)
        _ = [ax_wave.spines[w].set_linewidth(2) for w in ['top', 'bottom', 'left', 'right']]
        ax_wave.patch.set_alpha(0); fig_wave.patch.set_alpha(0)
        fig_wave.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        fig_spec, ax_spec = self.fig_spec, self.fig_spec.add_subplot()
        show_spec(audio_data, audio_fs, n_fft=2048, ax=ax_spec)
        # ax_spec.axis('off')
        _ = [ax_wave.spines[w].set_linewidth(2) for w in ['top', 'bottom', 'left', 'right']]
        fig_spec.patch.set_alpha(fig_wave.patch.get_alpha())
        fig_spec.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        self.update_wave_ticks()
        self.update_spec_ticks()

        wave_widget = FigureCanvasKivyAgg(fig_wave)
        spec_widget = FigureCanvasKivyAgg(fig_spec)
        dummy_widget = Widget()

        dummy_widget.height = '35sp'
        dummy_widget.size_hint_y = None
        t_widget = FigureCanvasKivyAgg(self.fig_t)
        t_widget.size = (self.timeline_width, dummy_widget.height)
        t_widget.size_hint = (None, None)
        y_widget = FigureCanvasKivyAgg(self.fig_y)
        f_widget = FigureCanvasKivyAgg(self.fig_f)

        wave_widget.width = spec_widget.width = self.timeline_width
        wave_widget.size_hint_x = spec_widget.size_hint_x = None

        seekbar = self.ids.seekbar

        self.ids.box_tl.clear_widgets()
        self.ids.box_yaxis.clear_widgets()

        self.ids.box_yaxis.add_widget(dummy_widget)
        self.ids.box_yaxis.add_widget(y_widget)
        self.ids.box_yaxis.add_widget(f_widget)

        self.ids.box_tl.add_widget(t_widget)
        self.ids.box_tl.add_widget(wave_widget)
        self.ids.box_tl.add_widget(spec_widget)
        self.ids.box_tl.add_widget(seekbar)

        if self.sound is None:
            self.sound = SoundLoader.load(audio_path)
        elif self.sound.source != audio_path:
            self.sound.source = audio_path
            self.sound.load()

        if self.audio_toolbar:
            self.sound.volume = self.audio_toolbar.ids.volume.value

            def on_value(instance, value):
                self.sound.volume = value

            self.audio_toolbar.ids.volume.unbind()
            self.audio_toolbar.ids.volume.bind(value=on_value)

        self.on_audio_pos(None, None)
        gc.collect()

    def update_wave_ticks(self):
        audio_data, audio_fs = self.audio_data, self.audio_fs
        ax_wave = self.fig_wave.axes[0]

        fig_t = self.fig_t
        fig_t.clear()
        ax_t = fig_t.add_subplot(sharex=ax_wave)
        ax_t.set_yticks([])
        ax_t.set_xlim(ax_wave.get_xlim())
        ax_t.minorticks_on()
        #ax_t.tick_params(which='both', axis='both', reset=True)
        ax_t.tick_params(
            which='both', axis='x', labelsize=self.label_size, length=self.majortick_length,
            top=True, labeltop=True, bottom=False, labelbottom=False,
        )
        ax_t.tick_params(
            which='both', axis='y', left=False, labelleft=False, right=False, labelright=False
        )
        ax_t.tick_params(which='minor', axis='x', length=2.5)
        ax_t.xaxis.set_major_formatter(ax_wave.xaxis.get_major_formatter())
        ax_t.patch.set_alpha(0)
        _ = [ax_t.spines[d].set_visible(False) for d in ['bottom', 'left', 'right']]
        fig_t.patch.set_alpha(self.fig_wave.patch.get_alpha())
        fig_t.subplots_adjust(left=0, right=1, bottom=0, top=.3, wspace=0, hspace=0)

        fig_y = self.fig_y
        fig_y.clear()
        ax_y = fig_y.add_subplot(sharey=ax_wave)
        ax_y.set_xticks([])
        #ax_y.tick_params(which='both', axis='both', reset=True)
        ax_y.tick_params(
            which='both', axis='x', top=False, labeltop=False, bottom=False, labelbottom=False
        )
        ax_y.tick_params(
            which='major', axis='y', labelsize=self.label_size, length=self.minortick_length,
            left=True, labelleft=True, right=False, labelright=False
        )
        ax_y.tick_params(which='minor', axis='y', left=False, right=False)
        _ = [ax_y.spines[d].set_visible(False) for d in ['top', 'bottom', 'right']]
        ax_y.patch.set_alpha(0)
        fig_y.patch.set_alpha(self.fig_wave.patch.get_alpha())
        fig_y.subplots_adjust(left=.85, right=1, bottom=0, top=1, wspace=0, hspace=0)

    def update_spec_ticks(self):
        ax_spec = self.fig_spec.axes[0]

        fig_f = self.fig_f
        fig_f.clear()
        ax_f = fig_f.add_subplot(sharey=ax_spec)
        ax_f.set_xticks([])
        ax_f.tick_params(
            which='both', axis='x',
            top=False, labeltop=False, bottom=False, labelbottom=False,
        )
        ax_f.tick_params(
            which='major', axis='y', labelsize=self.label_size, length=self.minortick_length,
            left=True, labelleft=True, right=False, labelright=False
        )
        ax_f.tick_params(which='minor', axis='y', left=False, right=False)
        ax_f.patch.set_alpha(0)
        _ = [ax_f.spines[d].set_visible(False) for d in ['top', 'bottom', 'right']]
        fig_f.patch.set_alpha(self.fig_spec.patch.get_alpha())
        fig_f.subplots_adjust(left=.85, right=1, bottom=0, top=1, wspace=0, hspace=0)

    def on_timeline_t_unit(self, instance, value):
        try:
            ax_t = self.fig_t.axes[0]
        except IndexError:
            return None

        t_unit = self.timeline_t_unit
        t_start, t_end = 0, int((self.audio_data.shape[-1]/self.audio_fs)/t_unit)*t_unit
        t_ticks = torch.arange(t_start, t_end+t_unit*.1, t_unit)

        ax_t.set_xticks(t_ticks)

        t_widget = FigureCanvasKivyAgg(self.fig_t)
        t_widget.size = (self.timeline_width, 70)
        t_widget.size_hint = (None, None)

        seekbar = self.ids.seekbar

        self.ids.box_tl.clear_widgets([seekbar, self.ids.box_tl.children[-1]])
        self.ids.box_tl.add_widget(t_widget, index=2)
        self.ids.box_tl.add_widget(seekbar)

        for child in self.ids.box_tl.children:
            child.width = self.timeline_width

    def on_audio_pos(self, instance, value):
        seekbar = self.ids.seekbar
        bar = seekbar.canvas.children[-1]

        if self.sound.length:
            bar.pos = (self.ids.box_tl.width*(self.audio_pos/self.sound.length), bar.pos[1])

    def init_timeline(self):
        _ = [fig.clear() for fig in [self.fig_wave, self.fig_spec]]

        if self.sound is not None:
            self.sound.unload()

        scrollview_width = self.ids.box_yaxis.parent.width
        audio_s = self.audio_data.size(-1)/self.audio_fs

        width_per_unit = 200

        maximum_unit_num = scrollview_width // width_per_unit

        t_unit = 2**int(audio_s**.5)
        while t_unit/2*maximum_unit_num > audio_s:
            t_unit *= 1/2

        self.timeline_t_unit = t_unit
        self.timeline_width = (audio_s*width_per_unit) // t_unit

class AudioToolbar(Container):
    check_pos = None
    root_audio_dict_container = None
    audio_timeline = None

    def on_kv_post(self, *arg, **kwargs):
        self.ch_window_menu = MDDropdownMenu(
            caller=self.ids.ch,
            width_mult=4,
            items=[dict(
                viewclass='OneLineListItem',
                text=f'{0:02d}ch',
                height=dp(54),
                on_release=lambda x=0: self.set_ch(x)
            )]
        )

    def set_ch(self, ch: int):
        if self.audio_timeline:
            if self.audio_timeline.audio_dict:
                if ch >= 0:
                    self.ids.ch.text = f'{ch:02d}ch'
                else:
                    self.ids.ch.text = 'mean'
            self.ch_window_menu.dismiss()

    def play(self):
        if self.audio_timeline:
            sound = self.audio_timeline.sound

            if sound and sound.state == 'stop':
                sound.seek(self.audio_timeline.audio_pos)
                sound.play()
                self.check_pos = Clock.schedule_interval(
                    lambda dt: self.position(), self.audio_timeline.check_dt
                )

    def pause(self):
        if self.audio_timeline:
            sound = self.audio_timeline.sound

            if sound and sound.state == 'play':
                sound.stop()
                Clock.unschedule(self.check_pos)
                self.check_pos = None

    def stop(self):
        if self.audio_timeline:
            sound = self.audio_timeline.sound

            if sound:
                sound.stop()
                self.position(0)
                Clock.unschedule(self.check_pos)
                self.check_pos = None
            else:
                self.position(0)

    def position(self, pos=None):
        if self.audio_timeline:
            sound = self.audio_timeline.sound

            if pos is None:
                self.audio_timeline.audio_pos = sound.get_pos()
            else:
                self.audio_timeline.audio_pos = pos

            if self.check_pos:
                if sound.length - self.audio_timeline.audio_pos <= 4*self.audio_timeline.check_dt:
                    self.audio_timeline.audio_pos = 0
                    return False

    def magnify(self, mode='plus'):
        if self.audio_timeline:
            seekbar = self.audio_timeline.ids.seekbar
            bar = seekbar.canvas.children[-1]
            bar_x = [bar.pos[0]][0]

            if not self.audio_timeline.audio_dict:
                return None

            # グラフの width の変化から拾ってほしいが，そっちだと上手く反映されないため，
            # 手動でバーの設定を行う
            if mode == 'plus':
                self.audio_timeline.timeline_width *= 2
                self.audio_timeline.timeline_t_unit /= 2

                bar.pos = (bar_x*2, bar.pos[1])

            elif mode == 'minus':
                self.audio_timeline.timeline_width /= 2
                self.audio_timeline.timeline_t_unit *= 2

                bar.pos = (bar_x/2, bar.pos[1])

    def close(self):
        if self.root_audio_dict_container:
            self.root_audio_dict_container.audio_dict = {}

    # def expand(self):
    #     if self.root_audio_dict_container and self.root_audio_dict_container.audio_dict:
    #         audio_dict = self.root_audio_dict_container.audio_dict
    #         audio_data, audio_fs = audio_dict['data'], audio_dict['fs']

    #         fig, axes = plt.subplots(2, 1)

class AudioMiniplot(FloatLayout):
    def __init__(self, *args, **kwargs):
        self.audio_data = kwargs.pop('data')
        self.audio_fs = kwargs.pop('fs')
        self.audio_path = kwargs.pop('path')
        self.figure = kwargs.pop('figure') if 'figure' in kwargs else None
        self.sound = kwargs.pop('sound') if 'sound' in kwargs else None

        super().__init__(*args, **kwargs)

        self.dt_sum = 0

        self.set_audio()

    def set_audio(self):
        self.set_plot()
        self.set_others()

    def set_plot(self):
        if self.figure is None:
            fig, axes = plt.subplots(2, 1)

            ax_wave, ax_spec = axes

            show_wave(self.audio_data, self.audio_fs, ax=ax_wave, color='b')
            show_spec(self.audio_data, self.audio_fs, ax=ax_spec, n_fft=2048)

            ax_wave.set_xlim(0, self.audio_data.size(-1)/self.audio_fs)
            ax_wave.tick_params(
                which='both', axis='both',
                top=True, labeltop=True, bottom=False, labelbottom=False,
                left=False, labelleft=False, right=False, labelright=False,
            )
            ax_wave.set_xlabel(''); ax_wave.set_ylabel('')

            ax_spec.tick_params(
                which='both', axis='both',
                top=False, labeltop=False, bottom=False, labelbottom=False,
                left=False, labelleft=False, right=False, labelright=False,
            )
            ax_spec.set_xlabel(''); ax_spec.set_ylabel('')

            fig.patch.set_alpha(0)
            _ = [ax.patch.set_alpha(0) for ax in axes]
            fig.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, wspace=0, hspace=0)

            self.figure = fig

        plot_widget = FigureCanvasKivyAgg(self.figure)
        plot_widget.size_hint = (.9, 1.)
        plot_widget.pos_hint = {'x': .05, 'y': 0.}

        self.add_widget(plot_widget)

    def set_others(self):
        if self.sound is None:
            self.sound = SoundLoader.load(self.audio_path)

        def play_button_clicked(x):
            if self.sound.source != self.audio_path:
                self.sound.source = self.audio_path
                self.sound.load()

            def change_progressbar_value(dt):
                self.dt_sum += dt
                self.progressbar.value = min(self.dt_sum/self.sound.length*100, 100)

                if self.progressbar.value == 100 or self.sound.source != self.audio_path:
                    self.progressbar.value, self.dt_sum = 0, 0
                    return False

            if self.sound.state == 'stop':
                self.sound.play()
                self.clock = Clock.schedule_interval(lambda dt: change_progressbar_value(dt), .5)
            elif self.sound.state == 'play':
                self.sound.stop()
                Clock.unschedule(self.clock)
                self.progressbar.value, self.dt_sum = 0, 0

        play_widget = MDFloatingActionButton(
            icon='play',
            type='small',
            theme_icon_color='Custom',
            md_bg_color=(1, 1, 1, .8),
            icon_color=(0, 0, 0, 1),
            pos_hint={'right': .95, 'y': .1},
            size_hint=(.2, .2),
            on_press=play_button_clicked
        )

        progressbar_widget = MDProgressBar(value=0)
        progressbar_widget.pos_hint = {'x': .05, 'y': .05}
        progressbar_widget.size_hint = (.9, .01)

        self.add_widget(play_widget)
        self.add_widget(progressbar_widget)

        self.progressbar = progressbar_widget

    # メモリリーク対策に clear_widgets 等の前に実行すること
    def reset(self):
        self.sound.unload()
        plt.close(self.figure)
        gc.collect()
