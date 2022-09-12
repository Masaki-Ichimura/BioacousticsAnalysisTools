from kivy.lang import Builder

from batools.app.gui.widgets.sub_tab import SubTab

Builder.load_file(__file__[:-3]+'.kv')


class GeneralTab(SubTab):
    pass
