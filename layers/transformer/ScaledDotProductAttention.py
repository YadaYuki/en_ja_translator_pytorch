import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    # TODO: masking
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scalar = np.sqrt(self.d_k)
        attention_weight = nn.functional.softmax(
            torch.matmul(q, torch.transpose(k, 1, 2)) / scalar, dim=2
        )
        return torch.matmul(attention_weight, v)
