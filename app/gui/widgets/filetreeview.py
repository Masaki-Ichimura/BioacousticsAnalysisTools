from kivy.lang import Builder
from kivy.uix.treeview import TreeView

Builder.load_file(__file__[:-3]+'.kv')


class FileTreeView(TreeView):
    def on_minimum_height(self, instance, value):
        self.height = self.minimum_height
