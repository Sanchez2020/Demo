import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid, PPI
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from texttable import Texttable
import seaborn as sns


def load_dataset(dataset_folder, dataset_name):
    """
    导入数据集，并处理为Data格式
    :param dataset_folder: 数据集存储路径
    :param dataset_name: 数据集的名字（"Cora", "CiteSeer", "PubMed"）
    :return: dataset
    """
    path = os.path.join(os.path.dirname(dataset_folder), dataset_name)
    dataset = Planetoid(path, dataset_name, T.NormalizeFeatures())
    return dataset


def load_PPI(dataset_folder):
    """
    导入PPI数据集，处理为Data格式并划分训练集，验证集，测试集
    :param dataset_folder: 数据集存储路径
    :return: 训练集，验证集，测试集
    """
    path = os.path.join(os.path.dirname(dataset_folder), 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

def test_PPI(dataset_folder):
    """
        导入PPI数据集，处理为Data格式
        :param dataset_folder: 数据集存储路径
        :return: 测试集
        """
    path = os.path.join(os.path.dirname(dataset_folder), 'PPI')
    test_dataset = PPI(path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_dataset, test_loader


def tab_printer(args):
    """
    使用表格形式打印日志
    :param args: 模型中使用的参数
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_accs(args, list_accs):
    """
    绘制训练过程中的准确率变化
    :param args: 参数
    :param list_accs: 变化率列表
    :return: svg格式的准确率变化图
    """
    print("正在绘图...")

    # 提取准确率
    train_accs = []
    val_accs = []
    test_accs = []

    # if args.model == "GAT_PPI":
    #     for i in range(len(list_accs)):
    #         train_accs.append(list_accs[i])
    #         val_accs.append(list_accs[i+1])
    #         test_accs.append(list_accs[i+2])
    # else:
    for l in list_accs:
        train_accs.append(l[0])
        val_accs.append(l[1])
        test_accs.append(l[2])

    # x的范围
    x = np.arange(1, args.epochs + 1)

    # 准确率
    tr_y = np.array(train_accs) * 100
    v_y = np.array(val_accs) * 100
    ts_y = np.array(test_accs) * 100

    plt.figure()

    # 坐标轴标签和字体设置
    plt.xlabel('epoch', fontsize=15)  # 坐标标签和字体设置
    if args.model == "GAT_PPI":
        plt.ylabel("Loss and F1(%)", fontsize=15)
    else:
        plt.ylabel("Accuracy(%)", fontsize=15)


    if args == "GAT_PPI":
        plt.title(args.model)
    else:
        plt.title(args.model + '_' + args.dataset_name)  # 标题

    plt.grid(linestyle='-.')  # 网格

    # 画图
    plt.plot(x, tr_y, label='train')
    plt.plot(x, v_y, color='red', label='validation')
    plt.plot(x, ts_y, color='green', label='test')

    # 图例
    if args.model == "GAT_PPI":
        plt.legend(labels=['train_loss', 'val_F1', 'test_F1'], loc='center right')
    else:
        plt.legend(labels=['train', 'validation', 'test'], loc='lower right')

    # 保存图片
    if args.model == "GAT_PPI":
        plt.savefig(args.result_path + args.model + '.svg', format='svg')
    else:
        plt.savefig(args.result_path + args.model + '_' + args.dataset_name + '_accs.svg', format='svg')

    plt.show()


def error_log():
    print("输入有误，请检查输入或重新输入！")


def scatter(x, colors, classes):
    """
    画点，用于嵌入可视化
    :param x: 嵌入结果
    :param colors: 颜色，每个label对应一种颜色
    :param classes: label类别数
    """
    palette = np.array(sns.color_palette("hls", classes))
    f = plt.figure(figsize=(5, 5))
    ax = plt.subplot()
    sc = ax.scatter(x[:, 0], x[:, 1], c=palette[colors.astype(np.int)])
    return f, ax, sc
