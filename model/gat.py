import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from scr.layers import ListModule


class GAT(torch.nn.Module):
    """
    GAT模型实现
    """

    def __init__(self, args, dataset, data):
        super(GAT, self).__init__()
        self.args = args
        self.dataset = dataset
        self.data = data
        self.define_model()

    def define_model(self):
        """
        定义神经网络模型
        """
        out_channels = self.args.attention_out_channels
        multi_head = self.args.multi_head
        dropout = self.args.dropout
        self.conv_layers = [GATConv(in_channels=self.dataset.num_features, out_channels=out_channels, heads=multi_head,
                                    dropout=dropout)]
        for layer in range(self.args.number_layers - 2):
            self.conv_layers.append(
                GATConv(in_channels=out_channels * multi_head, out_channels=out_channels, heads=multi_head,
                        dropout=dropout))
        self.conv_layers.append(
            GATConv(in_channels=out_channels * multi_head, out_channels=self.dataset.num_classes, dropout=dropout))
        self.conv_layers = ListModule(*self.conv_layers)

    def forward(self):
        """
        向前传播
        :return: out：结点分类结果，hidden_representations：嵌入表示
        """
        self.hidden_representations = []
        x = self.data.x
        for layer in self.conv_layers:
            x = F.dropout(x, self.args.dropout, training=self.training)
            x = F.elu(layer(x, self.data.edge_index))
            self.hidden_representations.append(x)
        out = F.log_softmax(x, dim=1)
        return out
