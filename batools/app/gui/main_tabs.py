from kivy.lang import Builder

from batools.app.gui.widgets.main_tab import MainTab

Builder.load_file(__file__[:-3]+'.kv')


class EditTab(MainTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class OffprocessTab(MainTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class ConfigTab(MainTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)