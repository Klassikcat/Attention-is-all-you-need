import torch
from torch import nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEncoding


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
        """
        Initializer of Encoder Module.
        """
        super(Encoder, self).__init__()
        self.positional_encoding = PositionalEncoding(
            input_dim, hidden_size=hidden_dim, is_pos_embed=is_positional_embedding, padding_idx=pad_idx
        )
        self.layers = nn.ModuleList(
            [
                MultiHeadAttention(hidden_dim, num_heads, dim_head) for _ in range(n_layers)
            ]
        )
        self.feed_forward = FeedForward(hidden_dim, feed_forward_expansion_factor, feed_forward_dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def _add_norm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.layernorm(torch.add(x, y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
            x = self.feed_forward(x)
        return x


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
                MultiHeadAttention(hidden_dim, num_heads, dim_head) for _ in range(n_layers)
            ]
        )
        self.feed_forward = FeedForward(hidden_dim, feed_forward_expansion_factor, feed_forward_dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensorl:
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
