import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class PPI(torch.nn.Module):
    def __init__(self, args, train_dataset):
        super(PPI, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.define_model()

    def define_model(self):
        out_channels = self.args.attention_out_channels
        multi_head = self.args.multi_head
        channels = multi_head * out_channels

        self.conv1 = GATConv(in_channels=self.train_dataset.num_features, out_channels=out_channels, heads=multi_head)
        self.lin1 = torch.nn.Linear(in_features=self.train_dataset.num_features, out_features=channels)
        self.conv2 = GATConv(in_channels=channels, out_channels=out_channels, heads=multi_head)
        self.lin2 = torch.nn.Linear(in_features=channels, out_features=channels)
        self.conv3 = GATConv(
            in_channels=channels, out_channels=self.train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(in_features=channels, out_features=self.train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        hidden_representations = x
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x, hidden_representations