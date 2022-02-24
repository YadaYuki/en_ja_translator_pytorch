import numpy as np
import torch
from torch import nn


class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoding_weight: torch.Tensor = self._initialize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_encoding_weight.unsqueeze(0)

    # TODO: refactoring, よりPytorch・numpyらしい書き方に変更する
    def _get_positional_encoding_val(self, pos: int, i: int) -> float:
        w = pos / (10000 ** (i / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight = [
            [
                self._get_positional_encoding_val(pos, i)
                for i in range(1, self.d_model + 1)
            ]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()
