from Trainer import GATTrainer, GCNTrainer
from scr.parser import parameter_parser
from scr.utils import tab_printer, draw_accs, error_log


def main():
    """
    测试
    """
    list_accs = []
    args = parameter_parser()
    tab_printer(args)
    if (args.model == "GAT"):
        model = GATTrainer(args)
    elif (args.model == "GCN"):
        model = GCNTrainer(args)
    else:
        error_log()
    list_accs = model.fit()
    draw_accs(args, list_accs)
    model.save_embedding()
    model.save_result()
    model.save_model()


if __name__ == "__main__":
    main()
