import torch
from module import *


def test_dot_product():
    query = torch.rand(3, 3, 3)
    key = torch.rand(3, 3, 3)
    value = torch.rand(3, 3, 3)
    mask = torch.rand(3, 3, 3)
    return scaled_dot_product_attention(query, key, value, mask, dropout_p=.1)


def test_positional_encoding():
    encoding = PositionalEncoding(501, hidden_size=128, padding_idx=0, is_pos_embed=True)
    x = torch.randint(1, 500, (1, 100))
    return encoding(x)


def test_multi_head_attention():
    return None


def main():
    assert test_dot_product()
    assert test_positional_encoding()
    assert test_multi_head_attention()


if __name__ == '__main__':
    main()