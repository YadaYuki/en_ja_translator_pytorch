# from os.path import join

import importlib

import numpy as np
import torch

# from torch.utils.data import DataLoader

# from const.path import TOK_CORPUS_PATH
# from layers.transformer.FFN import FFN


def pad_mask(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    mask = x.eq(0)  # 0 is <pad> in vocab
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1, seq_len, 1)
    return mask


if __name__ == "__main__":
    batch_size = 3
    seq_len = 3
    d_model = 3
    x = torch.randn(batch_size, seq_len, d_model).gt(0)
    y = torch.randn(batch_size, seq_len, d_model).gt(0)
    print(x.shape)
    print(x[:, :-1].shape)

    print(x)
    print(y)
    print(torch.logical_or(x, y))
