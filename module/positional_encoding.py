import torch
from torch import nn
import torch.nn.functional as F


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
            pos_info = F.embedding(positions, pos_info)   # [B, T] -> [B, T, D]
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
