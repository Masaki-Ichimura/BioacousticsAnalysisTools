from kivy.lang import Builder
from kivy.uix.widget import Widget

from app.gui.widgets.container import Container

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/config_container.kv')


class ConfigContainer(Container):
    def on_kv_post(self, *arg, **kwargs):
        self.ids.nav_drawer.set_state('open')
