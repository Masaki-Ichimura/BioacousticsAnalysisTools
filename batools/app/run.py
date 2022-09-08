from kivy.core.window import Window
from kivy.config import Config

import os
import matplotlib.style as mplstyle
import matplotlib as mpl

print(os.getcwd())

from batools.app.app import MainApp


def main():
    os.environ['KIVY_AUDIO'] = 'ffpyplayer'

    mplstyle.use('fast')
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 1.0
    mpl.rcParams['agg.path.chunksize'] = 10000
    # mpl.rcParams.update({'xtick.labelsize': 10, 'ytick.labelsize': 10})

    Window.size = 1440, 810
    Window.top, Window.left = 100, 100

    # マルチタッチ無効化
    Config.set('input', 'mouse', 'mouse, disable_multitouch')

    MainApp().run()


if __name__ == '__main__':
    main()