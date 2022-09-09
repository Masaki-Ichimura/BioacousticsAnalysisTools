from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup

Builder.load_file(__file__[:-3]+'.kv')


class FilechooserPopup(Popup):
    load = ObjectProperty()
