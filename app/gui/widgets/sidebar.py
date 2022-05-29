from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.treeview import TreeViewLabel

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/sidebar.kv')


class Sidebar(Widget):
    pass

class AudioTreeViewLabel(TreeViewLabel):
    pass
