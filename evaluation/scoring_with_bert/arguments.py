import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Image Classification', add_help=False)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--train_bs', default=8, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--adam_epsilon', default=1e-08, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int)
    parser.add_argument('--warmup_steps', default=20, type=int)
    parser.add_argument('--model', default='klue/roberta-small', type=str)
    parser.add_argument('--data', default='klue/sts', type=str)

    return parser