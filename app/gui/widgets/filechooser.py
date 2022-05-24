from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, StringProperty

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/filechooser.kv')


class FilechooserPopup(Popup):
    load = ObjectProperty()
