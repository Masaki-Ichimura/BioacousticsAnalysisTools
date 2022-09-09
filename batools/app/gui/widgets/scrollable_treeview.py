from kivy.lang import Builder
from kivy.uix.treeview import TreeView, TreeViewLabel

Builder.load_file(__file__[:-3]+'.kv')


class ScrollableTreeView(TreeView):
    pass

class AudioTreeViewLabel(TreeViewLabel):
    pass