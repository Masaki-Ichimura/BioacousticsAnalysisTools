from kivy.lang import Builder
from kivy.uix.widget import Widget

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/container.kv')


class Container(Widget):
    pass
