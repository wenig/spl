import torch
from torch import Tensor
import torch.nn as nn


class SPLLoss(nn.NLLLoss):
    def __init__(self, *args, n_samples=0, **kwargs):
        super(SPLLoss, self).__init__(*args, **kwargs)
        self.threshold = 0.1
        self.growing_factor = 1.3
        self.v = torch.zeros(n_samples).int()

    def forward(self, input: Tensor, target: Tensor, index: Tensor) -> Tensor:
        super_loss = nn.functional.nll_loss(input, target, reduction="none")
        v = self.spl_loss(super_loss)
        self.v[index] = v
        return (super_loss * v).mean()

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()
