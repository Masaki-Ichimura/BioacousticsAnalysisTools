from kivy.core.window import Window
from kivy.config import Config

from .app import MainApp


if __name__ == '__main__':
    Window.size = 1440, 810
    Window.top = 100
    Window.left = 100

    # マルチタッチ無効化
    Config.set('input', 'mouse', 'mouse, disable_multitouch')

    MainApp().run()
