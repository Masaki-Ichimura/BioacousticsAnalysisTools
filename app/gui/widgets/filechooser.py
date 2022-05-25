from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup


Builder.load_file('/'.join(__file__.split('/')[:-1])+'/filechooser.kv')


class FilechooserPopup(Popup):
    load = ObjectProperty()
