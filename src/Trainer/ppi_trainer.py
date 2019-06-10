import torch
from src.model.ppi_gat import Net
from src.utils import load_PPI, mkdir
from sklearn.metrics import f1_score

class PPITrainer(object):
    """
    GAT_PPI模型加载，训练，测试，保存结果。
    """

    def __init__(self, args):
        """
        :param args: 参数对象
        """
        self.args = args
        self.load_data()
        self.load_to_device()
        self.loss_op = torch.nn.BCEWithLogitsLoss()

    def load_data(self):
        """
        划分数据集并导入数据
        """
        self.train_dataset, self.val_dataset, self.test_dataset, self.train_loader, self.val_loader, self.test_loader = load_PPI(self.args.dataset_folder)

    def load_to_device(self):
        """
        加载模型到设备
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net(self.args, self.train_dataset).to(self.device)

    def train(self):
        """
        训练函数
        :return: 损失值
        """
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        total_loss = 0
        for data in self.train_loader:
            num_graphs = data.num_graphs
            data.batch = None
            data = data.to(self.device)
            optimizer.zero_grad()
            loss = self.loss_op(self.model(data.x, data.edge_index), data.y)
            total_loss += loss.item() * num_graphs
            loss.backward()
            optimizer.step()
        return total_loss / len(self.train_loader.dataset)

    def test(self, loader):
        """
        测试函数
        :param loader: 加载器
        """
        self.model.eval()

        ys, preds = [], []
        for data in loader:
            ys.append(data.y)
            with torch.no_grad():
                out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            preds.append((out > 0).float().cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

    def fit(self):
        """
        训练函数
        :return: F1分数
        """
        print("正在PPI数据集上训练GAT模型...")
        list_accs = []
        for epoch in range(1, self.args.epochs + 1):
            list_unit = []
            loss = self.train()
            list_unit.append(loss)
            val_acc = self.test(self.val_loader)
            list_unit.append(val_acc)
            test_acc = self.test(self.test_loader)
            list_unit.append(test_acc)
            list_accs.append(list_unit)
            print('epoch: {:02d}, loss: {:.4f}, val_f1: {:.4f}, test_f1: {:.4f}'.format(epoch, loss, val_acc, test_acc))
        return list_accs

    def save_model(self):
        """
        保存模型
        :return: 保存.pkl文件
        """
        mkdir(self.args.result_path)
        torch.save(self.model.state_dict(), self.args.result_path + 'GAT_PPI_model.pkl')

        print("模型已保存成功！")
