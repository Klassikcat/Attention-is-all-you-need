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
    Preform Dot-Product Attention proposed on Attention is All You need(2017)

    Dot-Product Attention calculates attention score by connecting all words in the given sentence to all words.
    There are three steps in forward propagation of dot-product attention.

        1.  Produce similarity score by calculating inner product of query and key.
        2.  Divide similarity score produced on above step by square root of the dimension of key.
            This normalizes the similarity score. Then, regularize the score by applying softmax function.
        3.  Finally, multiply regularized similarity score and value. This stage produces attention score.

    The concept of the dot-product attention is derived from time evolution(physics). To see the concept of the time
    evolution, see https://en.wikipedia.org/wiki/Time_evolution#:~:text=Time%20evolution%20is%20the%20change,be%20discrete%20or%20even%20finite.

    :param query: torch.Tensor, query matrix shapes [B, Q] (hidden states, decoder output, etc...]
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


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()