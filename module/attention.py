from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int = 0, is_pos_embed: bool = False):
        super(PositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.is_pos_embed = is_pos_embed

    def forward(self, inputs: torch.Tensor):
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

        ** Given B = Batch size, T = Sequence length, D = Dimension of embedding, K = Number of positions, **

        :param inputs: torch.Tensor, Input sequence tensors shapes [B, T].
        :return: torch.Tensor, Output Sequence tensors with positional information shapes [B, T, D].
        """
        n_seq = inputs.shape[1]      # Sequence Length
        #
        #   First, we embed the input matrix by applying embedding layer. this matrix will be used to calculate
        #   positional encoding in the last layer of this module.
        #
        x = self.embedding(inputs)   # [B, T] -> [B, T, D]
        #
        #   Next, we calculate positional encoding by applying sinusoid function.
        #   sinusoid function is a function applying sine and cosine functions depending on whether their index is
        #   odd or even. if their index is even, apply sine function, if odd, apply cosine function.
        #
        #   By using sine and cosine functions, we can inject positional information into the input matrix.
        #
        pos_info = self._get_sinusoid_table(n_seq)    # [B, T, D]
        if self.is_pos_embed:
            #
            #   Some architectures based on transformer utilize positional embedding. This is why we need to check
            #   whether the positional embedding is enabled or not.
            #
            #   There are three steps of positional embedding.
            #
            #   1. Get the position value having equal sized with the input sequence.
            #   2. Find padding value in the input. Then, change all padding values to zero in the positional value.
            #   3. get the embedding value of positional value by applying embedding layer.
            #
            #   Dimension of positional embedded tensor should be equal to the word embedded tensor.
            #
            positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
            pos_mask = inputs.eq(self.padding_idx)
            positions.masked_fill(pos_mask, 0)
            pos_info = torch.embedding(positions, pos_info)   # [B, T] -> [B, T, D]
            assert pos_info.shape == x.shape, \
                "Dimension of positional embedding tensor should be equal to the word embedding tensor."
        #
        #   Finally, we concatenate the positional embedding and word embedding.
        #
        return x + pos_info

    def _get_angle(self, pos: int, dim: int):
        """
        This function is used to calculate the angle of sinusoid function.
        """
        return pos / (10000 ** (2 * dim / self.hidden_size))

    def _get_sinusoid_table(self, n_seq: int):
        if self.is_pos_embed:
            n_seq = n_seq + 1
        sinusoid_table = torch.Tensor([[self._get_angle(pos, i_hidden) for i_hidden in range(self.hidden_size)] for pos in range(n_seq)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table


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


if __name__ == "__main__":
    encoding = PositionalEncoding(501, hidden_size=128, padding_idx=0, is_pos_embed=True)
    x = torch.randint(1, 500, (1, 100))
    print(encoding(x).shape)