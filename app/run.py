import os
os.environ['KIVY_AUDIO'] = 'ffpyplayer'

from kivy.core.window import Window
from kivy.config import Config

from .app import MainApp

import matplotlib.style as mplstyle
import matplotlib as mpl

mplstyle.use('fast')
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['agg.path.chunksize'] = 10000


if __name__ == '__main__':
    Window.size = 1440, 810
    Window.top = 100
    Window.left = 100

    # マルチタッチ無効化
    Config.set('input', 'mouse', 'mouse, disable_multitouch')

    MainApp().run()
