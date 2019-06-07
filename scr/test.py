from scr.Trainer import GATTrainer, GCNTrainer, PPITrainer
from scr.utils import draw_accs, error_log
from scr.gat_ppi_test import Test


def test_gat(args):
    """
    测试GAT模型
    :param args: 参数对象
    :return: 以csv格式保存测试结果，嵌入可视化
    """
    model = test(args)
    model.save_embedding()
    model.save_result()


def test_gcn(args):
    """
    测试GCN模型
    :param args: 参数对象
    :return: 以csv格式保存输出结果，嵌入可视化
    """
    model = test(args)
    model.save_embedding()
    model.save_result()


def test_ppi(args):
    """
    测试在PPI数据集上训练的GAT模型
    :param args: 参数对象
    :return: 以csv文件形式保存测试结果
    """
    model = test(args)
    ppi_test = Test(args)
    ppi_test.save()


def test(args):
    """
    测试训练模型
    :param args: 参数对象
    :return: 训练好的模型
    """
    # 结果变化列表
    list_accs = []

    # 选择模型
    if (args.model == "GAT_PPI"):
        model = PPITrainer(args)
    elif (args.model == "GAT"):
        model = GATTrainer(args)
    elif (args.model == "GCN"):
        model = GCNTrainer(args)
    else:
        error_log()

    # 训练模型
    list_accs = model.fit()

    # 训练过程可视化
    draw_accs(args, list_accs)

    # 保存模型
    model.save_model()
    return model
