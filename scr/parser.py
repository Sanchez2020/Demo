import argparse


def parameter_parser():
    """
    解析命令行参数
    """

    parser = argparse.ArgumentParser(description="Run GNN.")

    parser.add_argument("--dataset-folder",
                        nargs="?",
                        default="./input/",
                        help="Dataset path.")

    parser.add_argument("--dataset-name",
                        nargs="?",
                        default="Cora",
                        help="Dataset. Default is 'Cora'.")

    parser.add_argument("--result-path",
                        nargs="?",
                        default="./output/",
                        help="Path to store the result.")

    parser.add_argument("--model",
                        nargs="?",
                        default="GAT",
                        help="Model. Default is GAT.")

    parser.add_argument("--number-layers",
                        type=int,
                        default=2,
                        help="Number of GNN Layers. Default is 2.")

    parser.add_argument("--attention-out-channels",
                        type=int,
                        default=8,
                        help="Number of out channels of the GAT layer. Default is 8.")

    parser.add_argument("--multi-head",
                        type=int,
                        default=8,
                        help="Multi head. Default is 8.")

    parser.add_argument("--gcn-out-channels",
                        type=int,
                        default=64,
                        help="Number of out channels of the GCN layer. Default is 64.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.6,
                        help="Dropout. Default is 0.6.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.005,
                        help="Learning rate. Default is 0.005")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5e-4,
                        help="Weight decay. Default is 5*10^-4.")

    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of training epochs. Default is 200.")

    parser.add_argument("--n-iter",
                        type=int,
                        default=10000,
                        help="Number of iterations of the t-SNE algorithm. Default is 10000.")

    return parser.parse_args()
