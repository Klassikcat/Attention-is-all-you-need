import torch
import argparse
from typing import Dict
from omegaconf import DictConfig
import pytorch_lightning as pl
from module import Encoder, Decoder


def greedy_decode(state_dict: torch.nn, ):


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