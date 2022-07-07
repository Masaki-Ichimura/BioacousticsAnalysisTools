import torch

from kivy.properties import ObjectProperty


"""
    NOTE:
        - TorchTensor を格納するための Property
        - 以下を参考にした
        - https://qiita.com/kzm4269/items/3740c8957d8f5fcfeca7
"""
class TorchTensorWrapper(torch.Tensor):
    def __eq__(self, other):
        return torch.equal(self, other)

class TorchTensorProperty(ObjectProperty):
    def set(self, obj, val):
        if not isinstance(val, TorchTensorWrapper):
            val = val.as_subclass(TorchTensorWrapper)
        return super().set(obj, val)

    def get(self, *args, **kwargs):
        val = super().get(*args, **kwargs)
        return None if val is None else val.as_subclass(TorchTensorWrapper)
