from kivy.lang import Builder

from batools.app.gui.widgets.tab import Tab

Builder.load_file(__file__[:-3]+'.kv')


class GeneralTab(Tab):
    pass
