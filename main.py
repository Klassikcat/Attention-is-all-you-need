import torch
import argparse
from typing import Dict
from omegaconf import DictConfig
import pytorch_lightning as pl
from module import Encoder, Decoder


class TransformerModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super(TransformerModel, self).__init__()
        self.config = config
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.encoder = Encoder(
            config.encoder_input_dim,
            config.hidden_dim,
            config.num_heads,
            config.dim_head,
            config.encoder_ff_expansion_factor,
            config.encoder_n_layers,
            config.pad_idx,
            config.encoder_dropout,
            config.is_positional_embedding
        )
        self.decoder = Decoder(
            config.decoder_input_dim,
            config.hidden_dim,
            config.num_heads,
            config.dim_head,
            config.decoder_ff_expansion_factor,
            config.decoder_n_layers,
            config.pad_idx,
            config.decoder_dropout,
            config.is_positional_embedding
        )
        self.criterion = None

    def forward(self, encoder_input: torch.Tensor):
        encoder_output, encoder_attn_probs = self.encoder(encoder_input)
        target_seq = torch.LongTensor([self.sos_id])
        decoder_output, decoder_attn_probs, enc_dec_attn_probs = self.decoder(encoder_output, encoder_attn_probs)

    def configure_optimizers(self) -> Dict[str: torch.optim.Optimizer, str: torch.optim.lr_scheduler]:
        self.optimizers = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.lr_scheduler = None


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