from typing import Optional

import numpy as np
import torch
from torch import nn


def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: Optional[float] = None):
    """
    Perform Dot-Product Attention proposed on Attention is All You need(2017)

    Dot-Product Attention calculates attention score by connecting all words in the given sentence to all words.
    There are three steps in forward propagation of dot-product attention.

        1.  Produce similarity score by calculating inner product of query and key.
        2.  Divide similarity score produced on above step by square root of the dimension of key.
            This normalizes the similarity score. Then, regularize the score by applying softmax function.
        3.  Finally, multiply regularized similarity score and value. This stage produces attention score.

    The concept of the dot-product attention is derived from time evolution(physics). To see the concept of the time
    evolution, see https://en.wikipedia.org/wiki/Time_evolution#:~:text=Time%20evolution%20is%20the%20change,be%20discrete%20or%20even%20finite.

    :param query: torch.Tensor, query matrix shapes [B, 1, Q] (hidden states, decoder output, etc...]
    :param key: torch.Tensor, key matrix shapes [T, B, K] (encoder outputs)
    :param value: torch.Tensor, value matrix shapes [T, B, V] (encoder outputs)
    :param mask: torch.Tensor, mask location shapes.
    :param dropout_p: float between 0 and 1.
    :return: Tuple of torch.Tensor, Attention Score shapes [B, V] and regularized similarity score shapes [B, 1, T]
    """

    #   First, perform dot-product to produce similarity score between key and query.
    #   Inner product between query and key matrix will give the similarity score matrix
    #   between all elements of query and key matrix.

    sim = torch.matmul(query, torch.transpose(key, 1, 2))  # [B, 1, Q] * [B, K, T] = [B, 1, T]

    #   Next, Normalize all similarity score by dividing square root of the dimension.
    #   Then, regularize normalized score by applying softmax function.
    #   Apply dropout if you need.

    scale = 1 / np.sqrt(key.shape[1])
    normalized_sim = sim * scale   # [B, 1, T]
    if mask:
        normalized_sim = normalized_sim.masked_fill_(mask, -1e-9)   # Mask Added
    regularlized_sim = torch.softmax(normalized_sim, dim = -1)      # Regularized

    if dropout_p:
        regularlized_sim = torch.dropout(regularlized_sim, dropout_p, train=torch.is_grad_enabled())

    #   Finally, multiply value and regularized similarity score and Value.
    value = value.transpose(0, 1)   # [T, B, V] -> [B, T, V]
    attention_score = torch.matmul(regularlized_sim, value)     # [B, 1, T] * [B, T, V] -> [B, V]

    return attention_score, regularlized_sim


def positional_encoding(inputs: torch.Tensor):
    """
    Forward propagation of positional encoding.

    This method embeds input matrix with positional representation by apply sine and cosine function.

    Since Transformer model does not contain convolutional subsampling or recurrence module, relative or absolute
    positional information should be injected into the input matrix in order to the model make use of the order of the
    sequences. There are two methods to inject positional information -- absolute and relative. There are many ways to
    inject positional information to the sequence matrix such as learned or fixed, so that is up you.

    usually, this functions added into bottom of the encoder and decoder stacks. However, some architectures such as
    conformer stacks onto the module itself.

    In this method, we use sine and cosine functions of different frequencies shown in the below.

    PE_(pos,2i) = sin(pos/10000^2i/d_model)
    PE_(pos,2i+1) = cos(pos/10000^2i/d_model)

    where pos is the position and i is the dimension.

    That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric
    progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily
    learn to attend by relative positions, since for any fixed offset k, P Epos+k can be represented as a linear
    function of PE_pos.

    :param inputs: torch.Tensor, Input sequence tensors.
    :return: torch.Tensor, Output Sequence tensors with positional information.
    """


class MultiHeadAttention(nn.Module):
    """
    initializer of Multi-Head Attention
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def forward(self, x: torch.Tensor):
        """
        Forward propagation of Multi-Head Attention.

        Multi-head attention calculates multiple attention scores, then concat all attention scores to get a result.

        There are three reasons of why use Self-Attention.
            1.  Computational complexity per layer.
            2.  Amount of computation that can be parallelized, as measured by the minimum number of sequential
                operations required.
            3.  the path length between long-range dependencies in the network. One key factor affecting the ability to
                learn such dependencies is the length of the paths forward and backward signals have to traverse in the
                network. The shorter these paths between any combination of positions in the input and output sequences,
                the easier it is to learn long-range dependencies. Hence we also compare the maximum path length between
                any two input and output positions in networks composed of the different layer types.

        :param x:
        :return:
        """