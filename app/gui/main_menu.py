from kivy.uix.widget import Widget
from kivy.lang import Builder

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/main_menu.kv')


class MainMenuWidget(Widget):
    pass
