import torch


class ListModule(torch.nn.Module):
    """
    抽象层列表类
    """

    def __init__(self, *args):
        """
        模型初始化
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        获取层索引
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        迭代
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        层数
        """
        return len(self._modules)
