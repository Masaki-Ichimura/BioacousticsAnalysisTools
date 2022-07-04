import torch

from kivy.lang import Builder

from app.gui.widgets.tab import Tab

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/preprocessed.kv')


class PreprocessedTab(Tab):
    pass
