import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        self.W_k = nn.Parameter(
            torch.Tensor(h, d_model, self.d_k)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

        self.W_q = nn.Parameter(
            torch.Tensor(h, d_model, self.d_k)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

        self.W_v = nn.Parameter(
            torch.Tensor(h, d_model, self.d_v)  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        return q, k
