import argparse


def parameter_parser(args):
    """
    解析命令行参数
    """

    # 解析器
    parser = argparse.ArgumentParser(prog='Demo', description="Run GNN.")

    # 输入路径
    parser.add_argument("--dataset-folder",
                        nargs="?",
                        type=str,
                        default="./input/",
                        help="Dataset path.")

    # 输出路径
    parser.add_argument("--result-path",
                        nargs="?",
                        type=str,
                        default="./output/",
                        help="Path to store the result.")

    # 子解析器
    subparsers = parser.add_subparsers(title="sub-commands",
                                       dest="model",
                                       description="valid sub-commands",
                                       help="sub-command help.")

    # GAT模型解析器
    parser_gat = subparsers.add_parser("GAT", help="GAT help.")

    # 数据集
    parser_gat.add_argument("--dataset-name",
                            nargs="?",
                            type=str,
                            choices=["Cora", "CiteSeer", "PubMed"],
                            default="Cora",
                            help="Dataset. Default is 'Cora'.")

    parser_gat.add_argument("--number-layers",
                            type=int,
                            default=2,
                            help="Number of GAT Layers. Default is 2.")

    parser_gat.add_argument("--attention-out-channels",
                            type=int,
                            default=8,
                            help="Number of out channels of the GAT layer. Default is 8.")

    parser_gat.add_argument("--multi-head",
                            type=int,
                            default=8,
                            help="Multi head. Default is 8.")

    parser_gat.add_argument("--dropout",
                            type=float,
                            default=0.6,
                            help="Dropout. Default is 0.6.")

    parser_gat.add_argument("--learning-rate",
                            type=float,
                            default=0.005,
                            help="Learning rate. Default is 0.005.")

    parser_gat.add_argument("--weight-decay",
                            type=float,
                            default=5e-4,
                            help="Weight decay. Default is 5*10^-4.")

    parser_gat.add_argument("--epochs",
                            type=int,
                            default=200,
                            help="Number of training epochs. Default is 200.")

    parser_gat.add_argument("--n-iter",
                            type=int,
                            default=10000,
                            help="Number of iterations of the t-SNE algorithm. Default is 10000.")

    # GCN模型解析器
    parser_gcn = subparsers.add_parser("GCN", help="GCN help.")

    # 数据集
    parser_gcn.add_argument("--dataset-name",
                            nargs="?",
                            type=str,
                            choices=["Cora", "CiteSeer", "PubMed"],
                            default="PubMed",
                            help="Dataset. Default is 'Cora'.")

    parser_gcn.add_argument("--number-layers",
                            type=int,
                            default=2,
                            help="Number of GCN Layers. Default is 2.")

    parser_gcn.add_argument("--gcn-out-channels",
                            type=int,
                            default=64,
                            help="Number of out channels of the GCN layer. Default is 64.")

    parser_gcn.add_argument("--learning-rate",
                            type=float,
                            default=0.01,
                            help="Learning rate. Default is 0.01.")

    parser_gcn.add_argument("--weight-decay",
                            type=float,
                            default=5e-4,
                            help="Weight decay. Default is 5*10^-4.")

    parser_gcn.add_argument("--epochs",
                            type=int,
                            default=200,
                            help="Number of training epochs. Default is 200.")

    parser_gcn.add_argument("--n-iter",
                            type=int,
                            default=10000,
                            help="Number of iterations of the t-SNE algorithm. Default is 10000.")

    # PPI解析器
    parser_ppi = subparsers.add_parser("GAT_PPI", help="PPI help.")

    parser_ppi.add_argument("--dataset-name",
                            nargs="?",
                            type=str,
                            choices=["PPI"],
                            default="PPI",
                            help="Dataset. Default is 'PPI'.")

    parser_ppi.add_argument("--number-layers",
                            type=int,
                            default=3,
                            choices=[3],
                            help="Number of GAT Layers. Default is 3.")

    parser_ppi.add_argument("--attention-out-channels",
                            type=int,
                            default=256,
                            help="Number of out channels of the GAT layer. Default is 256.")

    parser_ppi.add_argument("--multi-head",
                            type=int,
                            default=4,
                            help="Multi head. Default is 4.")

    parser_ppi.add_argument("--learning-rate",
                            type=float,
                            default=0.005,
                            help="Learning rate. Default is 0.005.")

    parser_ppi.add_argument("--epochs",
                            type=int,
                            default=100,
                            help="Number of training epochs. Default is 100.")

    return parser.parse_args(args)
