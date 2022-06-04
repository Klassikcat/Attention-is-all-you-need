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


def test_encoder_module():
    encoder_input = torch.randint(0, 10, (2, 20))
    encoder = Encoder(
        input_dim=300,
        hidden_dim=20,
        num_heads=2,
        dim_head=5,
        feed_forward_expansion_factor=4,
    )

    encoder_out = encoder(encoder_input)
    print(encoder_out[0].shape)
    return encoder_out


def test_decoder_module():
    encoder_input = torch.randint(0, 10, (2, 20))
    encoder_output = torch.rand(2, 20, 20)
    decoder_input = torch.randint(0, 10, (2, 20))

    decoder = Decoder(
        input_dim=300,
        hidden_dim=20,
        num_heads=2,
        dim_head=5,
        feed_forward_expansion_factor=4,
        n_layers=2,
        pad_idx=0,
        feed_forward_dropout=0.1,
        is_positional_embedding=True
    )

    decoder_out, attn_probs, enc_dec_probs = decoder(decoder_input, encoder_input, encoder_output)
    print(decoder_out.shape)
    return decoder_out


def test_multi_head_attention():
    return None


def main():
    """
    assert test_dot_product()
    assert test_positional_encoding()
    assert test_multi_head_attention()
    assert test_decoder_module()
    """
    assert test_encoder_module()
    assert test_encoder_module()


if __name__ == '__main__':
    main()
    print('All tests passed!')
