import torch
import torch.nn.functional as F
from model.gat import GAT
from utils import load_dataset, mkdir, scatter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class GATTrainer(object):
    """
    GAT模型加载，训练，测试，保存结果。
    """

    def __init__(self, args):
        """
        :param args: 参数对象
        """
        self.args = args
        self.dataset = load_dataset(self.args.dataset_folder, self.args.dataset_name)
        self.data = self.dataset[0]
        self.load_to_device()

    def load_to_device(self):
        """
        加载数据和模型到设备
        :return:
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GAT(self.args, self.dataset, self.data).to(device)
        self.data = self.data.to(device)

    def train(self):
        """
        训练函数
        :return:
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        F.nll_loss(self.model()[self.data.train_mask], self.data.y[self.data.train_mask]).backward()
        optimizer.step()

    def test(self):
        """
        测试函数
        :return:
        """
        self.model.eval()
        logits, accs = self.model(), []
        for _, mask in self.data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    def fit(self):
        """
        训练过程
        :return:
        """
        list_accs = []
        for epoch in range(1, self.args.epochs + 1):
            self.train()
            list_accs.append(self.test())
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, *self.test()))
        return list_accs

    def save_model(self):
        """
        保存模型
        :return:
        """
        mkdir(self.args.result_path)
        torch.save(self.model.state_dict(), self.args.result_path + 'gat_model.pkl')

    def save_result(self):
        """
        保存节点预测结果
        :return: 以csv文件存入文件
        """
        mkdir(self.args.result_path)
        pre = self.model()[self.data.test_mask].max(1)[1].view(1000, 1)
        real = self.data.y[self.data.test_mask].view(1000, 1)
        result = torch.cat((pre, real), 1).detach().cpu().numpy()
        index = ["node_" + str(x) for x in range(1000)]
        columns = ["prediction", "real"]
        result = pd.DataFrame(result, index=index, columns=columns)
        result.to_csv(self.args.result_path + self.args.model + '_' + self.args.dataset_name + '_result.csv',
                      index=None)

    def save_embedding(self):
        """
        保存嵌入结果
        :return: 以csv格式存入文件
        """
        mkdir(self.args.result_path)
        embedding = self.model.hidden_representations[-2][self.data.test_mask].detach().cpu().numpy()

        X = embedding
        Y = self.data.y[self.data.test_mask].detach().cpu().numpy()
        GATTrainer.embed_visualization(self, X=X, Y=Y)
        index = ["node_" + str(x) for x in range(1000)]
        columns = ["x_" + str(x) for x in range(len(embedding[0]))]
        embedding = pd.DataFrame(embedding, index=index, columns=columns)
        embedding.to_csv(self.args.result_path + self.args.model + '_' + self.args.dataset_name + '_embedding.csv',
                         index=None)

    def embed_visualization(self, X, Y):
        """
        :param X: 节点嵌入结果
        :param Y: 节点标签
        :return: 嵌入的降维可视化结果
        """
        tsne = TSNE(n_components=2, init='pca', n_iter=self.args.n_iter)
        X_tsne = tsne.fit_transform(X)
        c = self.dataset.num_classes
        scatter(X_tsne, Y, c)
        plt.savefig(self.args.result_path + '/' + self.args.model + '_' + self.args.dataset_name + '_embedding.svg',
                    format='svg')
        plt.show()
