import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from texttable import Texttable
import seaborn as sns


def load_dataset(dataset_folder, dataset_name):
    """
    导入数据集
    :param dataset_folder: 数据集存储路径
    :param dataset_name: 数据集的名字（"Cora", "CiteSeer", "PubMed"）
    :return: dataset
    """
    path = os.path.join(os.path.dirname(dataset_folder), dataset_name)
    dataset = Planetoid(path, dataset_name, T.NormalizeFeatures())
    return dataset


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
    # 提取准确率
    train_accs = []
    test_accs = []
    val_accs = []

    for l in list_accs:
        train_accs.append(l[0])
        test_accs.append(l[1])
        val_accs.append(l[2])

    # x的范围
    x = np.arange(1, args.epochs + 1)

    # 准确率
    tr_y = np.array(train_accs) * 100
    ts_y = np.array(test_accs) * 100
    v_y = np.array(val_accs) * 100

    plt.figure()

    # 坐标轴标签和字体设置
    plt.xlabel('epoch', fontsize=15)  # 坐标标签和字体设置
    plt.ylabel("Accuracy(%)", fontsize=15)

    plt.title(args.model + '_' + args.dataset_name)  # 标题

    plt.grid(linestyle='-.')  # 网格

    # 画图
    plt.plot(x, tr_y, label='train')
    plt.plot(x, ts_y, color='red', label='test')
    plt.plot(x, v_y, color='green', label='validation')

    # 图例
    plt.legend(labels=['train', 'test', 'validation'], loc='lower right')

    # 保存图片
    plt.savefig(args.result_path + '/' + args.model + '_' + args.dataset_name + '_accs.svg', format='svg')

    plt.show()


def error_log():
    print("输入有误，请检查输入或重新输入！")


def scatter(x, colors, classes):
    """
    画点，用于嵌入可视化
    :param x: 嵌入结果
    :param colors: 颜色，每个label对应一种颜色
    """
    palette = np.array(sns.color_palette("hls", classes))
    f = plt.figure(figsize=(5, 5))
    ax = plt.subplot()
    sc = ax.scatter(x[:, 0], x[:, 1], c=palette[colors.astype(np.int)])
    return f, ax, sc
