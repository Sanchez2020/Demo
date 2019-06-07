from scr.parser import parameter_parser
from scr.utils import tab_printer, error_log
from scr.test import test_gat, test_gcn, test_ppi


def main(args):
    """
    测试
    """
    # 解析参数
    args = parameter_parser(args)

    # 打印参数
    tab_printer(args)

    # 选择模型
    if (args.model == "GAT_PPI"):
        test_ppi(args)
    elif (args.model == "GAT"):
        test_gat(args)
    elif (args.model == "GCN"):
        test_gcn(args)
    else:
        error_log()


if __name__ == "__main__":

    # 参数列表
    args = ["GAT_PPI"]
    main(args)
