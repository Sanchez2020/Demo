import torch
from src.utils import test_PPI, mkdir
from src.model import PPI
import pandas as pd


class Test(object):
    """
    测试PPI数据集上训练的GAT模型
    """
    def __init__(self, args):
        """
        :param args: 参数对象
        """
        # 参数
        self.args = args

        # 导入数据
        self.test_dataset, self.test_loader = test_PPI(self.args.dataset_folder)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 定义模型
        self.model = PPI(self.args, self.test_dataset).to(self.device)

        # 加载模型
        self.model.load_state_dict(torch.load(self.args.result_path + 'GAT_PPI_model.pkl'))

    def test(self, loader):
        """
        测试函数
        :param loader: 加载器
        :return: 真实值列表，预测值列表，嵌入表示列表
        """
        self.model.eval()
        ys, preds, emds = [], [], []
        for data in loader:
            ys.append(data.y)
            with torch.no_grad():
                reault, embedding = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            preds.append((reault > 0).float().cpu())
            emds.append(embedding)
        return ys, preds, emds

    def save(self):
        """
        保存测试结果
        :return: csv文件
        """
        reals, preds, emds = self.test(self.test_loader)
        self.save_reals(reals)
        self.save_preds(preds)
        self.save_embedding(emds)

    def save_reals(self, reals):
        """
        保存真实值
        :param reals: 测试集的真实值列表
        :return: 如：real_1.csv
        """
        print("正在保存测试集的真实值...")
        mkdir(self.args.result_path)
        count = 1
        for o in reals:
            result = pd.DataFrame(o.cpu().numpy())
            result.to_csv(self.args.result_path + 'real_' + str(count) + '.csv', index=None)
            count += 1
        print("测试集的真实值保存成功！")

    def save_preds(self, preds):
        """
        保存预测值
        :param preds: 测试集的预测值列表
        :return: 如：pred_1.csv
        """
        print("正在保存测试集的预测值...")
        mkdir(self.args.result_path)
        count = 1
        for o in preds:
            result = pd.DataFrame(o.cpu().numpy())
            result.to_csv(self.args.result_path + 'pred_' + str(count) + '.csv', index=None)
            count += 1
        print("测试集的预测值保存成功！")


    def save_embedding(self, emds):
        """
        保存嵌入结果
        :param emds: 测试集的嵌入表示列表
        :return: 如：embedding_1.csv
        """
        print("正在保存嵌入结果...")
        mkdir(self.args.result_path)
        count = 1
        for o in emds:
            result = pd.DataFrame(o.cpu().numpy())
            result.to_csv(self.args.result_path + 'embedding_' + str(count) + '.csv', index=None)
            count += 1
        print("嵌入结果保存成功！")
