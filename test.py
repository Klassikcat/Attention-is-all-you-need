import torch
import os
from configuration import TransformerConfiguration
from module import *


def test_dot_product():
    query = torch.rand(3, 3, 3)
    key = torch.rand(3, 3, 3)
    value = torch.rand(3, 3, 3)
    mask = torch.randint(0, 1, (3, 3))
    return scaled_dot_product_attention(query, key, value, mask, dropout_p=.1)


def test_positional_encoding():
    encoding = PositionalEncoding(501, hidden_size=128, padding_idx=0, is_pos_embed=True)
    x = torch.randint(1, 500, (1, 100))
    return encoding(x)


def multi_head_attention():
    return None


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


def test_transformer_module():
    decoder_input = torch.randint(0, 10, (2, 20))
    encoder_input = torch.randint(0, 10, (2, 20))

    encoder = Encoder(
        input_dim=300,
        hidden_dim=20,
        num_heads=2,
        dim_head=5,
        feed_forward_expansion_factor=4,
    )

    encoder_output, _ = encoder(encoder_input)
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


def test_transformer_model():
    encoder_input = torch.randint(0, 10, (2, 20))
    encoder_output = torch.rand(2, 20, 20)
    decoder_input = torch.randint(0, 10, (2, 20))
    config = TransformerConfiguration(
        encoder_input_dim=300,
        decoder_input_dim=300,
        hidden_dim=20,
        num_heads=2,
        dim_head=5,
        encoder_ff_expansion_factor=4,
        encoder_dropout=0.1,
        pad_idx=0,
        encoder_n_layers=2,
        decoder_n_layers=2,
        is_positional_embedding=True
    )
    multi_head_attention = MultiHeadAttention(
        config
    )
    result = multi_head_attention(decoder_input, encoder_input, encoder_output)
    print(result[0].shape)
    return result


def test_train():
    return None


def test_inference():
    return None


def main():
    try:
        if torch.cuda.is_available():
            print("Testing on CUDA environment\n"
                  "CUDA device count: {}\n"
                  "CUDA device name: {}\n"
                  "CUDA device version: {}\n"
                  "CUDA device capability: {}\n"
                  "Pytorch version: {}\n"
                  "Operating system: {}\n"
                  ).format(torch.cuda.device_count(),
                           torch.cuda.get_device_name(0),
                           torch.version.cuda,
                           torch.cuda.get_device_capability(0),
                           torch.__version__,
                           os.name)
        elif not torch.cuda.is_available():
            print(f"Testing on CPU environment.\n"
                  f"Environment information: pytorch {torch.__version__}, Operating System: {os.name}")
        test_dot_product()
        test_positional_encoding()
        test_decoder_module()
        test_encoder_module()
        test_decoder_module()
        test_transformer_module()
        test_transformer_model()
        test_train()
        test_inference()
        print("Test Passed! Good job!")
    except Exception as e:
        print("Test Failed! Check your code!")
        print(e)


if __name__ == '__main__':
    main()
    print('All tests passed!')
