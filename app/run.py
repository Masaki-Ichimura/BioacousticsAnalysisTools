from kivy.core.window import Window
from .app import MainApp


if __name__ == '__main__':
    Window.size = 1440, 810
    Window.top = 100
    Window.left = 100

    MainApp().run()
