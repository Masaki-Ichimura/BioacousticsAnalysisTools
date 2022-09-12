from kivy.lang import Builder
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.tab import MDTabsBase

Builder.load_file(__file__[:-3]+'.kv')


class SubTab(MDFloatLayout, MDTabsBase):
    pass
