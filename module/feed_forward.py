import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: int, dropout_p=0.2):
        """
        initializer of FeedForward class
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        """
        Forward propagation of Feed Forward Network of Transformer Network.

        Feed Forward network is a basic network that consists of two dense layer. This is the primitive form of the
        neural network which only goes forward. While other networks, such as recurrent network and convolutional netwo-
        rk have special structure such as recurrent structure or convolutional calculation, this network does not have
        any special structure.

        Args:
        :param hidden_dim: int, hidden dim of the Feed-Forward network.
        :param expansion_factor: int, expansion factor of the Feed-Forward network.
        :param dropout_p: float, dropout probability of the Feed-Forward network.

        Given B = batch size, D = hidden dim, E = expansion factor, and T = sequence length,

        Input:
        :x: torch.Tensor, input tensor of the Feed-Forward network shapes [B, T, D].
        :return: torch.Tensor shapes [B, T, D]
        """
        out = self.fc1(x.transpose(1, 2))           # [B, T, D] -> [B, D * E, T]
        out = self.relu(out)                        # [B, D * E, T] -> [B, D * E, T]
        out = self.fc2(out).transpose(1, 2)         # [B, D * E, T] -> [B, T, D]
        out = self.dropout(out)                     # [B, T, D] -> [B, T, D]
        return out
