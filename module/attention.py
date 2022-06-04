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
    Forward propagation of Dot-Product Attention proposed in Attention is All You need(2017)

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

    sim = torch.matmul(query, torch.transpose(key, -1, -2))  # [B, K, T, H] * [B, K, T, H] = [B, K, T, H]

    #   Next, Normalize all similarity score by dividing square root of the dimension.
    #   Then, regularize normalized score by applying softmax function.
    #   Apply dropout if you need.

    scale = 1 / np.sqrt(key.shape[1])
    normalized_sim = sim * scale  # [B, 1, T]
    if mask is not None:
        normalized_sim = normalized_sim.masked_fill_(mask, -1e-9)  # Mask Added
    regularlized_sim = torch.softmax(normalized_sim, dim=-1)  # Regularized

    if dropout_p:
        regularlized_sim = torch.dropout(regularlized_sim, dropout_p, train=torch.is_grad_enabled())

    #   Finally, multiply value and regularized similarity score and Value.
    value = value.transpose(0, 1)  # [T, B, V] -> [B, T, V]
    attention_score = torch.matmul(regularlized_sim, value)  # [B, 1, T] * [B, T, V] -> [B, V]

    return attention_score, regularlized_sim


def encoder_attn_mask(query: torch.Tensor, key: torch.Tensor, pad_idx: int):
    """
    Masking encoder padding positions.

    :param query: torch.Tensor, query matrix shapes [B, 1, Q] (hidden states, decoder output, etc...]
    :param key: torch.Tensor, key matrix shapes [T, B, K] (encoder outputs)
    :param pad_idx: torch.Tensor, pad index
    :return: torch.Tensor, mask location shapes.
    """
    batch_size, query_len = query.size()
    _, key_len = key.size()
    pad_attn_mask = key.data.eq(pad_idx)
    return pad_attn_mask.unsqueeze(1).expand(batch_size, query_len, key_len)


def decoder_attn_mask(seq: torch.Tensor):
    """
    Masking decoder padding positions.

    :param seq: torch.Tensor, sequence matrix shapes [B, 1, Q] (hidden states, decoder output, etc...]
    :return: torch.Tensor, mask location shapes.
    """
    # Create mask
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1)  # upper triangular part of a matrix(2-D)
    return subsequent_mask


class MultiHeadAttention(nn.Module):
    """
    initializer of Multi-Head Attention
    """

    def __init__(self, hidden_dim: int, num_heads: int, dim_head: int):
        super(MultiHeadAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, dim_head * num_heads)
        self.key_proj = nn.Linear(hidden_dim, dim_head * num_heads)
        self.value_proj = nn.Linear(hidden_dim, dim_head * num_heads)
        self.dense = nn.Linear(dim_head * num_heads, hidden_dim)
        self.num_heads = num_heads
        self.dim_head = dim_head

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
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

        Given B = batch_size, T = seq_len, D = hidden_dim, H = num_heads, K = dim_head

        :param x: torch.Tensor, shapes [B, T, D]
        :return:
        """

        # get batch_size, sequence length, hidden dim.

        batch_size = query.shape[0]
        seq_len = query.shape[1]
        hidden_dim = query.shape[2]

        #
        # First, Split x value into query, key, and value, then reshape them into [B, H, T, D * H]
        # Then, Split query, key and value into multiple heads to perform multi-head attention.
        #
        # Step 1. split x into query, key and value.
        #

        query = self.query_proj(query)  # [B, T, D] -> [B, T, K * H]
        key = self.key_proj(key)  # [B, T, D] -> [B, T, K * H]
        value = self.value_proj(value)  # [B, T, D] -> [B, T, K * H]

        #
        # Step 2. split query, key and value into multiple heads.
        #

        query = query.view(batch_size, -1, self.num_heads, self.dim_head)  # [B, T, K * H] -> [B, T, K, H]
        key = key.view(batch_size, -1, self.num_heads, self.dim_head)  # [B, T, K * H] -> [B, T, K, H]
        value = value.view(batch_size, -1, self.num_heads, self.dim_head)  # [B, T, K * H] -> [B, T, K, H]

        #
        #   Step 3. Change shape of tensors [B, T, K, H] into [B, K, T, H].
        #

        query = query.transpose(1, 2)  # [B, T, K, H] -> [B, K, H, T]
        key = key.transpose(1, 2)  # [B, T, K, H] -> [B, K, H, T]
        value = value.transpose(1, 2)  # [B, T, K, H] -> [B, K, H, T]

        # if masks exists, perform same steps to masks.
        if mask is not None:
            mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        #
        #   Next, calculate attention score of the all heads.
        #

        context, attention_score = scaled_dot_product_attention(query, key, value, mask)  # [B, K, T, H] -> [B, K, T, H]

        #
        #   Step 4. Change shape of tensors [B, K, T, H] into [B, T, K * H].
        #
        #   1. [B, K, H, T] -> [B, H, K, T]
        #   2. [B, H, K, T] -> [B, T, K * H]
        #

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head)
        context = self.dense(context)   # [B, T, K * H] -> [B, T, D]
        return context, attention_score
