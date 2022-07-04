from kivy.lang import Builder
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.tab import MDTabsBase

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/tab.kv')


class Tab(MDFloatLayout, MDTabsBase):
    pass
