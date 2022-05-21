import argparse


def train(args):
    raise NotImplementedError


def inference(args):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='train phase')
    parser.add_argument('--inference', action='store_false',
                        help='inference phase')
    parser.add_argument('--learning_rate', type=float, required=False,
                        help='learning rate of the model.')