from typing import Tuple, List
from omegaconf import DictConfig

import torch
from torch import nn
from .attention import MultiHeadAttention, encoder_attn_mask, decoder_attn_mask
from .feed_forward import FeedForward
from .positional_encoding import PositionalEncoding


class EncoderModule(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dim_head: int,
            feed_forward_expansion_factor: int,
            feed_forward_dropout: float = 0.5,
    ):
        """
        Initializer of Encoder Module.
        """
        super(EncoderModule, self).__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dim_head)
        self.feed_forward = FeedForward(hidden_dim, feed_forward_expansion_factor, feed_forward_dropout)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation of Encoder Module.

        Args:
            :param hidden_dim: hidden dimension of Encoder Module.
            :param num_heads: number of heads of MultiHeadAttention.
            :param dim_head: dimension of head of MultiHeadAttention.
            :param feed_forward_expansion_factor: expansion factor of FeedForward.
            :param feed_forward_dropout: dropout rate of FeedForward.

        Params:
            Given Batch size = B, Sequence Length = T
            :param x: input tensor of Encoder Module shapes [B, H, T].

        Returns:
            Given Batch size = B, Sequence Length = T, Hidden Dimension = H
            :return: output tensor of Encoder Module shapes [B, H, T]
        """
        attn_output, attn_prob = self.attention(x, x, x, mask)
        attn_output = self.layernorm1(attn_output + x)
        ffn_output = self.feed_forward(attn_output)
        layernorm_output = self.layernorm2(attn_output + ffn_output)
        return layernorm_output, attn_prob


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_heads: int,
            dim_head: int,
            feed_forward_expansion_factor: int,
            n_layers: int = 1,
            pad_idx: int = 0,
            feed_forward_dropout: float = 0.5,
            is_positional_embedding: bool = True,
    ):
        super(Encoder, self).__init__()
        self.positional_encoding = PositionalEncoding(
            input_dim, hidden_size=hidden_dim, is_pos_embed=is_positional_embedding, padding_idx=pad_idx
        )
        self.attn_modules = nn.ModuleList([
            EncoderModule(
                hidden_dim, num_heads, dim_head, feed_forward_expansion_factor, feed_forward_dropout
            ) for _ in range(n_layers)
        ])
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """
        Forward propagation of Encoder Module.

        Args:
            :param input_dim: input dimension of Encoder Module.
            :param hidden_dim: hidden dimension of Encoder Module.
            :param num_heads: number of heads of MultiHeadAttention.
            :param dim_head: dimension of head of MultiHeadAttention.
            :param feed_forward_expansion_factor: expansion factor of FeedForward.
            :param n_layers: number of layers of Encoder Module.
            :param pad_idx: padding index of Encoder Module.
            :param feed_forward_dropout: dropout rate of FeedForward.
            :param is_positional_embedding: whether to use positional embedding or not.

        Params:
            Given Batch size = B, Sequence Length = T
            :param x: input tensor of Encoder Module shapes [B, T].

        Returns:
            Given Batch size = B, Sequence Length = T, Hidden Dimension = H
            :return: output tensor of Encoder Module shapes [B, H, T]
        """
        encoder_output = self.positional_encoding(x)
        attn_mask = encoder_attn_mask(x, x, self.pad_idx)
        attn_probs = list()
        for module in self.attn_modules:
            module_output, attn_prob = module(encoder_output, attn_mask)
            attn_probs.append(attn_prob)
        return module_output, attn_probs


class DecoderModule(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dim_head: int,
            feed_forward_expansion_factor: int,
            feed_forward_dropout: float,
    ):
        """
        initializer of DecoderModule
        """
        super(DecoderModule, self).__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dim_head)
        self.encoder_attention = MultiHeadAttention(hidden_dim, num_heads, dim_head)
        self.feed_forward = FeedForward(hidden_dim, feed_forward_expansion_factor, feed_forward_dropout)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.layernorm3 = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor, encoder_inputs: torch.Tensor, decoder_mask: torch.Tensor, enc_dec_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward propagation of Decoder Module.

        Args:
        :param hidden_dim: hidden dimension of Encoder Module.
        :param num_heads: number of heads of MultiHeadAttention.
        :param dim_head: dimension of head of MultiHeadAttention.
        :param feed_forward_expansion_factor: expansion factor of FeedForward.
        :param feed_forward_dropout: dropout rate of FeedForward.

        Params:
        Given Batch size = B, Sequence Length = T
        :param x: input tensor of Encoder Module shapes [B, T, H].
        :param encoder_output: output tensor of Encoder Module shapes [B, T, H].

        Returns:
        Given Batch size = B, Sequence Length = T, Hidden Dimension = H
        :return: output tensor of Encoder Module shapes [B, T, H], encoder attention probability tensor shapes
        [B, n_heads, T, T], and encoder-decoder attention probability tensor shapes [B, n_heads, T, T].
        """
        #
        # [B, T, H] -> [B, T, H], [B, n_heads, T, T]
        #
        attention_output, attn_probs = self.attention(inputs, inputs, inputs, decoder_mask)
        #
        #   [B, T, H] -> [B, T, H]
        #
        layernorm_output = self.layernorm1(attention_output + inputs)
        #
        #   [B, T, H], [B, T, H] -> [B, T, H], [B, n_heads, T, T]
        #
        enc_dec_output, enc_dec_attn_probs = self.encoder_attention(layernorm_output, encoder_inputs, encoder_inputs, enc_dec_mask)
        #
        #   [B, T, H] -> [B, T, H]
        #
        enc_dec_output = self.layernorm2(attention_output + enc_dec_output)
        #
        #   [B, T, H] -> [B, T, H]
        #
        ffn_output = self.feed_forward(enc_dec_output)
        #
        #   [B, T, H] -> [B, T, H]
        #
        x = self.layernorm3(ffn_output + enc_dec_output)
        #
        #   :returns: [B, T, H], [B, n_heads, T, T], [B, n_heads, T, T]
        #
        return x, attn_probs, enc_dec_attn_probs


class Decoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_heads: int,
            dim_head: int,
            feed_forward_expansion_factor: int,
            n_layers: int = 1,
            pad_idx: int = 0,
            feed_forward_dropout: float = 0.5,
            is_positional_embedding: bool = True,
    ):
        """
        Initializer of Decoder Module.
        """
        super(Decoder, self).__init__()
        self.positional_encoding = PositionalEncoding(
            input_dim, hidden_size=hidden_dim, is_pos_embed=is_positional_embedding, padding_idx=pad_idx
        )
        self.layers = nn.ModuleList(
            [
                DecoderModule(
                    hidden_dim, num_heads, dim_head, feed_forward_expansion_factor, feed_forward_dropout
                ) for _ in range(n_layers)
            ]
        )
        self.pad_idx = pad_idx

    def forward(self, decoder_inputs: torch.Tensor, encoder_inputs: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, List, List]:
        """
        Forward propagation of Decoder Module.

        Args:
            :param input_dim: input dimension of Encoder Module.
            :param hidden_dim: hidden dimension of Encoder Module.
            :param num_heads: number of heads of MultiHeadAttention.
            :param dim_head: dimension of head of MultiHeadAttention.
            :param feed_forward_expansion_factor: expansion factor of FeedForward.
            :param n_layers: number of layers of Encoder Module.
            :param pad_idx: padding index of Encoder Module.
            :param feed_forward_dropout: dropout rate of FeedForward.
            :param is_positional_embedding: whether to use positional embedding or not.

        Params:
            Given Batch size = B, Sequence Length = T
            :param x: input tensor of Encoder Module shapes [B, T].
            :param encoder_output: output tensor of Encoder Module shapes [B, H, T].

        Returns:
            Given Batch size = B, Sequence Length = T, Hidden Dimension = H
            :return: output tensor of Encoder Module shapes [B, H, T]
        """
        decoder_output = self.positional_encoding(decoder_inputs)
        decoder_pad_masks = encoder_attn_mask(decoder_inputs, decoder_inputs, self.pad_idx)
        decoder_attn_masks = decoder_attn_mask(decoder_inputs)
        dec_self_attn_mask = torch.gt((decoder_pad_masks + decoder_attn_masks), 0)
        # (bs, n_dec_seq, n_enc_seq)
        dec_enc_attn_mask = encoder_attn_mask(decoder_inputs, encoder_inputs, self.pad_idx)
        attn_probs, enc_dec_attn_probs = list(), list()
        for layer in self.layers:
            decoder_output, attention_prob, enc_dec_attn_prob = layer(
                decoder_output, encoder_outputs, dec_self_attn_mask, dec_enc_attn_mask
            )
            attn_probs.append(attention_prob), enc_dec_attn_probs.append(enc_dec_attn_prob)
        return decoder_output, attn_probs, enc_dec_attn_probs


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super(Seq2SeqTransformer, self).__init__()
        self.encoder = Encoder(
            config.input_dim,
            config.hidden_dim,
            config.num_heads,
            config.dim_head,
            config.encoder_feed_forward_expansion_factor,
            config.encoder_n_layers,
            config.pad_idx,
            config.feed_forward_dropout,
            config.is_positional_embedding,

        )
        self.decoder = Decoder(
            config.input_dim,
            config.hidden_dim,
            config.num_heads,
            config.dim_head,
            config.decoder_feed_forward_expansion_factor,
            config.decoder_n_layers,
            config.pad_idx,
            config.feed_forward_dropout,
            config.is_positional_embedding,
        )

    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor) -> Tuple[torch.Tensor, list, list, list]:
        encoder_output, enc_attn_probs = self.encoder(encoder_inputs)
        decoder_outputs, dec_attn_probs, enc_dec_attn_probs = self.decoder(decoder_inputs, encoder_inputs, encoder_output)
        return decoder_outputs, enc_attn_probs, dec_attn_probs, enc_dec_attn_probs

    def encode(self, input_x: torch.Tensor, input_mask: torch.Tensor):
        return self.encoder(input_x, input_mask)

    def decode(self, input_y: torch.Tensor, memory: torch.Tensor, input_mask: torch.Tensor):
        return self.decoder(input_y, memory, input_mask)

