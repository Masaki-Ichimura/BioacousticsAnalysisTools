from kivy.lang import Builder

from batools.app.gui.widgets.container import Container

Builder.load_file(__file__[:-3]+'.kv')


class MainContainer(Container):
    pass
