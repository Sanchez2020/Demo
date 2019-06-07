import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scr.layers import ListModule


class GCN(torch.nn.Module):
    """
    GCN模型实现
    """
    def __init__(self, args, dataset, data):
        super(GCN, self).__init__()
        self.args = args
        self.dataset = dataset
        self.data = data
        self.define_model()

    def define_model(self):
        """
        定义神经网络模型
        """
        out_channels = self.args.gcn_out_channels
        self.conv_layers = [GCNConv(in_channels=self.dataset.num_features, out_channels=out_channels, cached=True)]
        for layer in range(self.args.number_layers - 2):
            self.conv_layers.append(GCNConv(in_channels=out_channels, out_channels=out_channels, cached=True))
        self.conv_layers.append(GCNConv(in_channels=out_channels, out_channels=self.dataset.num_classes, cached=True))
        self.conv_layers = ListModule(*self.conv_layers)

    def forward(self):
        """
        向前传播
        :return: out：结点分类结果，hidden_representations：嵌入表示
        """
        self.hidden_representations = []
        x = self.data.x
        count = 1
        for layer in self.conv_layers:
            if (count < len(self.conv_layers)):
                x = F.relu(layer(x, self.data.edge_index))
                x = F.dropout(x, training=self.training)
            count += 1
        x = self.conv_layers[len(self.conv_layers) - 1](x, self.data.edge_index)
        self.hidden_representations.append(x)
        out = F.log_softmax(x, dim=1)
        return out
