from typing import Optional

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', ignore_index: Optional[int] = None):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Forward propagation of the CrossEntropy loss.

        The negative log likelihood loss. It is useful to train a classification
        problem with `C` classes.
        If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
        weight to each of the classes. This is particularly useful when you have an
        unbalanced training set.
        The `input` given through a forward call is expected to contain
        log-probabilities of each class. `input` has to be a Tensor of size either
        :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
        with :math:`K \geq 1` for the `K`-dimensional case (described later).
        Obtaining log-probabilities in a neural network is easily achieved by
        adding a  `LogSoftmax`  layer in the last layer of your network.
        You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
        layer.
        The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
        where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
        this class index (this index may not necessarily be in the class range).
        The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
        .. math::
            \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
            l_n = - w_{y_n} x_{n,y_n}, \quad
            w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\},
        where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
        :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
        (default ``'mean'``), then
        .. math::
            \ell(x, y) = \begin{cases}
                \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
                \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}
        Can also be used for higher dimension inputs, such as 2D images, by providing
        an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
        where :math:`K` is the number of dimensions, and a target of appropriate shape
        (see below). In the case of images, it computes NLL loss per-pixel.
        Args:
            reduction: str, type of the reduction applied to the output. Default: ``'mean'``
            ignore_index (int, optional): Specifies a target value that is ignored
                and does not contribute to the input gradient. When :attr:`size_average` is
                ``True``, the loss is averaged over non-ignored targets. Default: None
        Inputs: logits, targets
            - logits (torch.FloatTensor): probability distribution value from model and it has a logarithm shape.
                The `FloatTensor` of size ``(batch, seq_length, num_classes)``
            - targets (torch.LongTensor): ground-truth encoded to integers which directly point a word in label.
                The `LongTensor` of size ``(batch, target_length)``
        Returns: loss
            * loss (float): loss for training
        Examples::
            >>> B, T1, C, T2 = 3, 128, 4, 10
            >>> loss = CrossEntropyLoss()
            >>> inputs = torch.randn(B, T1, C, requires_grad=True)
            >>> targets = torch.empty(B, T2, dtype=torch.long).random_(T2)
            >>> outputs = loss(inputs, targets)
            >>> outputs.backward()
        """

        logits = logits.contiguous().view(-1, logits.size(-1))

        return self.cross_entropy_loss(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1),
        )


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, reduction: str = 'mean', ignore_index: Optional[int] = None, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = logits.contiguous().view(-1, logits.size(-1))

        return self.criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1),
        )
