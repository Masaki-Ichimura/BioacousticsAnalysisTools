import matplotlib.pyplot as plt

from kivy.lang import Builder
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from kivy.uix.widget import Widget
from kivy.garden.matplotlib import FigureCanvasKivyAgg

from app.gui.main_container import MainContainer
from utils.audio import load_wav
from utils.plot import show_spec, show_wav

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/edit_container.kv')


class EditContainer(MainContainer):
    pass

class EditWorkingContainer(MainContainer):
    audio_file = StringProperty('')
    audio_data = None
    audio_fs = NumericProperty(0)

    def on_audio_file(self, instance, value):
        audio_toolbar = self.ids.audio_display.ids.audio_toolbar
        audio_timeline = self.ids.audio_display.ids.audio_timeline
        audio_toolbar.set_audio()

        audio_data, audio_fs = load_wav(self.audio_file)

        fig, axes = plt.subplots(2, 1, tight_layout=True)
        show_wav(audio_data, audio_fs, ax=axes[0])
        axes[0].set_xlim(0, audio_data.shape[-1]/audio_fs)
        axes[0].tick_params(
            bottom=False, labelbottom=False, left=False, labelleft=False
        )
        show_spec(audio_data, audio_fs, ax=axes[1])
        axes[1].tick_params(
            bottom=False, labelbottom=False, left=False, labelleft=False
        )
        axes[1].xaxis.set_visible(False); axes[1].yaxis.set_visible(False)

        timeline_widget = FigureCanvasKivyAgg(fig)

        audio_timeline.ids.box.clear_widgets()
        audio_timeline.ids.box.add_widget(timeline_widget)

        # fig.savefig('img.png')

        self.audio_data, self.audio_fs = audio_data, audio_fs

class EditAudioDisplay(MainContainer):
    pass

class EditAudioDetail(MainContainer):
    pass
