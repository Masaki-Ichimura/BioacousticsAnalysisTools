from kivy.lang import Builder
from kivy.uix.treeview import TreeView

Builder.load_file('/'.join(__file__.split('/')[:-1])+'/filetreeview.kv')


class FileTreeView(TreeView):
    def on_minimum_height(self, instance, value):
        self.height = self.minimum_height
