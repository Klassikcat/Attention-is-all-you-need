import torch
from src import *


def test_dot_product():
    query = torch.rand(3, 3, 3)
    key = torch.rand(3, 3, 3)
    value = torch.rand(3, 3, 3)
    mask = torch.rand(3, 3, 3)
    return scaled_dot_product_attention(query, key, value, mask, dropout_p=.1)