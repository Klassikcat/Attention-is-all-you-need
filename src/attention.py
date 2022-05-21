import torch
from torch import nn


class DotProductAttention(nn.Module):
    """
    Dot-Product Attention initializer.
    """
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Preform Dot-Product Attention proposed on Attention is All You need(2017)

        Dot-Product Attention calculates attention score by connecting all words in the given sentence to all words.
        There are three steps in forward propagation of dot-product attention.

            1.  Produce similarity score by calculating inner product of query and key.
            2.  Divide similarity score produced on above step by squared dimension of key.
                This normalizes the similarity score. Then, regularize the score by applying softmax function.
            3.  Finally, multiply regularized similarity score and value. This stage produces attention score.

        The concept of the dot-product attention is derived from time evolution(physics). To see the concept of the time
        evolution, see https://en.wikipedia.org/wiki/Time_evolution#:~:text=Time%20evolution%20is%20the%20change,be%20discrete%20or%20even%20finite.

        :param query: torch.Tensor, query matrix
        :param key: torch.Tensor, key matrix
        :param value: torch.Tensor, value matrix
        :return: torch.Tensor, Attention Score.
        """
        mat = torch.dot(query, key)     # First,


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()