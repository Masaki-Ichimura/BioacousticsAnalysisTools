import torch
import matplotlib.pyplot as plt

from kivy.lang import Builder
from kivy.properties import StringProperty, ObjectProperty, NumericProperty, BooleanProperty, Clock
from kivy.uix.widget import Widget
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivy.core.audio import SoundLoader

from app.gui.main_container import MainContainer
from utils.plot import show_spec, show_wav

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/audiotimeline.kv')


class AudioTimeline(MainContainer):
    audio_file = StringProperty('')

    audio_data = None
    Audio_fs = None

    fig_wav = ObjectProperty(None)
    fig_spec = ObjectProperty(None)

    timeline_t_unit = NumericProperty(1.)
    timeline_width = NumericProperty(1000)

    sound = ObjectProperty(None)
    audio_pos = NumericProperty(0)

    check_dt = .05

    def on_audio_file(self, instance, value):
        working_container = self.parent.parent.parent.parent
        audio_data = working_container.audio_data
        audio_fs = working_container.audio_fs

        self.audio_data, self.audio_fs = audio_data, audio_fs

        self.init_timeline()

        t_unit = self.timeline_t_unit
        t_start = 0
        t_end = int((audio_data.shape[-1]/audio_fs)/t_unit)*t_unit
        t_ticks = torch.arange(t_start, t_end+t_unit*.1, t_unit)

        fig_wav, ax_wav = plt.subplots(tight_layout=True)
        show_wav(audio_data, audio_fs, ax=ax_wav, color='b')
        ax_wav.set_xticks(t_ticks)
        ax_wav.set_xlim(0, audio_data.shape[-1]/audio_fs)
        ax_wav.patch.set_alpha(0)
        ax_wav.tick_params(
            which='both', axis='both', labelsize=15, length=0,
            top=False, labeltop=False, bottom=False, labelbottom=False,
            left=False, labelleft=False, right=False, labelright=False
        )
        ax_wav.set_xlabel(''); ax_wav.set_ylabel('')
        fig_wav.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig_wav.patch.set_alpha(0)

        fig_spec, ax_spec = plt.subplots(tight_layout=True)
        show_spec(audio_data, audio_fs, n_fft=2048, ax=ax_spec)
        ax_spec.tick_params(
            which='both', axis='both', labelsize=15, length=0,
            top=False, labeltop=False, bottom=False, labelbottom=False,
            left=False, labelleft=False, right=False, labelright=False
        )
        ax_spec.tick_params(axis='y', labelsize=10)
        ax_spec.set_xlabel(''); ax_spec.set_ylabel('')
        fig_spec.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig_spec.patch.set_alpha(fig_wav.patch.get_alpha())

        self.fig_wav, self.fig_spec = fig_wav, fig_spec

        wav_widget = FigureCanvasKivyAgg(self.fig_wav)
        spec_widget = FigureCanvasKivyAgg(self.fig_spec)
        dummy_widget = Widget()

        dummy_widget.height = 80
        dummy_widget.size_hint_y = None
        t_widget = FigureCanvasKivyAgg(self.fig_t)
        t_widget.size = (self.timeline_width, 80)
        t_widget.size_hint = (None, None)
        y_widget = FigureCanvasKivyAgg(self.fig_y)
        f_widget = FigureCanvasKivyAgg(self.fig_f)

        wav_widget.width = self.timeline_width
        wav_widget.size_hint_x = None
        spec_widget.width = self.timeline_width
        spec_widget.size_hint_x = None

        seekbar = self.ids.box_tl.children[0]
        self.ids.box_tl.clear_widgets()
        self.ids.box_yaxis.clear_widgets()

        self.ids.box_yaxis.add_widget(dummy_widget)
        self.ids.box_yaxis.add_widget(y_widget)
        self.ids.box_yaxis.add_widget(f_widget)
        self.ids.box_tl.add_widget(t_widget)
        self.ids.box_tl.add_widget(wav_widget)
        self.ids.box_tl.add_widget(spec_widget)
        self.ids.box_tl.add_widget(seekbar)

        self.sound = SoundLoader.load(self.audio_file)
        self.on_audio_pos(None, None)

    def on_fig_wav(self, instance, value):
        audio_data, audio_fs = self.audio_data, self.audio_fs
        ax_wav = self.fig_wav.axes[0]

        fig_t, ax_t = plt.subplots(tight_layout=True)
        ax_t.set_xlim(ax_wav.get_xlim())
        ax_t.set_xticks(ax_wav.get_xticks())
        ax_t.minorticks_on()
        ax_t.xaxis.set_major_formatter(ax_wav.xaxis.get_major_formatter())
        ax_t.patch.set_alpha(0)
        ax_t.tick_params(
            which='both', axis='both', reset=True
        )
        ax_t.tick_params(
            which='both', axis='y', labelsize=15, labelcolor=(0,0,0,0),
            left=False, labelleft=False, right=False, labelright=False
        )
        ax_t.tick_params(
            which='both', axis='x', labelsize=15, length=10,
            top=True, labeltop=True, bottom=False, labelbottom=False,
        )
        ax_t.tick_params(
            which='minor', axis='x', length=5
        )
        ax_t.spines['bottom'].set_visible(False)
        ax_t.spines['right'].set_visible(False)
        ax_t.spines['left'].set_visible(False)
        fig_t.patch.set_alpha(self.fig_wav.patch.get_alpha())

        fig_y = plt.figure()
        ax_y = fig_y.add_subplot(sharey=ax_wav)
        ax_y.patch.set_alpha(0)
        ax_y.tick_params(
            which='both', axis='both', reset=True
        )
        ax_y.tick_params(
            which='both', axis='x', labelsize=15, labelcolor=(0,0,0,0),
            top=False, labeltop=False, bottom=False, labelbottom=False,
        )
        ax_y.tick_params(
            which='major', axis='y', labelsize=12, length=5,
            left=True, labelleft=True, right=False, labelright=False
        )
        ax_y.tick_params(
            which='minor', axis='y', left=False, right=False
        )
        ax_y.spines['top'].set_visible(False)
        ax_y.spines['bottom'].set_visible(False)
        ax_y.spines['right'].set_visible(False)
        fig_y.patch.set_alpha(self.fig_wav.patch.get_alpha())

        fig_y.subplots_adjust(left=.85)
        self.fig_t, self.fig_y = fig_t, fig_y

    def on_fig_spec(self, instance, value):
        ax_spec = self.fig_spec.axes[0]

        fig_f = plt.figure()
        ax_f = fig_f.add_subplot(sharey=ax_spec)
        ax_f.patch.set_alpha(0)
        ax_f.tick_params(
            which='both', axis='both', reset=True
        )
        ax_f.tick_params(
            which='both', axis='x', labelsize=15, labelcolor=(0,0,0,0),
            top=False, labeltop=False, bottom=False, labelbottom=False,
        )
        ax_f.tick_params(
            which='major', axis='y', labelsize=10, length=5,
            left=True, labelleft=True, right=False, labelright=False
        )
        ax_f.tick_params(
            which='minor', axis='y', left=False, right=False
        )
        ax_f.spines['top'].set_visible(False)
        ax_f.spines['bottom'].set_visible(False)
        ax_f.spines['right'].set_visible(False)
        fig_f.patch.set_alpha(self.fig_spec.patch.get_alpha())

        # fig_f.tight_layout()
        fig_f.subplots_adjust(left=.85)
        self.fig_f = fig_f

    def on_timeline_t_unit(self, instance, value):
        ax_t = self.fig_t.axes[0]

        t_unit = self.timeline_t_unit
        t_start = 0
        t_end = int((self.audio_data.shape[-1]/self.audio_fs)/t_unit)*t_unit
        t_ticks = torch.arange(t_start, t_end+t_unit*.1, t_unit)

        ax_t.set_xticks(t_ticks)
        self.fig_t.subplots_adjust(left=0, right=1, bottom=0, top=1)

        t_widget = FigureCanvasKivyAgg(self.fig_t)
        t_widget.size = (self.timeline_width, 80)
        t_widget.size_hint = (None, None)

        seekbar = self.ids.box_tl.children[0]

        self.ids.box_tl.clear_widgets([seekbar, self.ids.box_tl.children[-1]])
        self.ids.box_tl.add_widget(t_widget, index=2)
        self.ids.box_tl.add_widget(seekbar)

        for child in self.ids.box_tl.children:
            child.width = self.timeline_width

    def init_timeline(self):
        scrollview_width = self.ids.box_yaxis.parent.width
        audio_s = self.audio_data.shape[-1]/self.audio_fs

        width_per_unit = 210

        maximum_unit_num = scrollview_width//width_per_unit

        t_unit = 2**int(audio_s**(1/2))
        while t_unit/2*maximum_unit_num>audio_s:
            t_unit *= 1/2

        self.timeline_t_unit = t_unit
        self.timeline_width = int(audio_s/t_unit*width_per_unit)

    def on_audio_pos(self, instance, value):
        seekbar = self.ids.seekbar
        bar = seekbar.canvas.children[-1]
        bar.pos = (
            self.ids.box_tl.width*(self.audio_pos/self.sound.length)+15,
            bar.pos[1]
        )

class AudioToolbar(MainContainer):
    check_pos = None

    def play(self):
        audio_timeline = self.parent.parent.ids.audio_timeline
        sound = audio_timeline.sound

        if sound and sound.state == 'stop':
            sound.seek(audio_timeline.audio_pos)
            sound.play()
            self.check_pos = Clock.schedule_interval(
                lambda dt: self.position(), audio_timeline.check_dt
            )

    def pause(self):
        audio_timeline = self.parent.parent.ids.audio_timeline
        sound = audio_timeline.sound

        if sound and sound.state == 'play':
            sound.stop()
            Clock.unschedule(self.check_pos)
            self.check_pos = None

    def stop(self):
        audio_timeline = self.parent.parent.ids.audio_timeline
        sound = audio_timeline.sound

        if sound:
            sound.stop()
            self.position(0)
            Clock.unschedule(self.check_pos)
            self.check_pos = None
        else:
            self.position(0)

    def position(self, pos=None):
        audio_timeline = self.parent.parent.ids.audio_timeline
        sound = audio_timeline.sound

        if pos is None:
            audio_timeline.audio_pos = sound.get_pos()
        else:
            audio_timeline.audio_pos = pos

        if self.check_pos:
            if sound.length - audio_timeline.audio_pos <= 4*audio_timeline.check_dt:
                audio_timeline.audio_pos = 0
                return False

    def magnify(self, mode='plus'):
        audio_timeline = self.parent.parent.ids.audio_timeline
        seekbar = audio_timeline.ids.seekbar
        bar = seekbar.canvas.children[-1]
        bar_x = [bar.pos[0]][0]

        # グラフの width の変化から拾ってほしいが，そっちだと上手く反映されないため，
        # 手動でバーの設定を行う
        if mode == 'plus':
            audio_timeline.timeline_width *= 2
            audio_timeline.timeline_t_unit /= 2

            bar.pos = (bar_x*2, bar.pos[1])

        elif mode == 'minus':
            audio_timeline.timeline_width /= 2
            audio_timeline.timeline_t_unit *= 2

            bar.pos = (bar_x/2, bar.pos[1])
