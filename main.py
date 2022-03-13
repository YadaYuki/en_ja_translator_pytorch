# # from os.path import join

# import importlib

# import numpy as np
import torch
from torch import nn

from layers.transformer.TransformerDecoder import TransformerDecoder


def pad_mask(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    mask = x.eq(0)  # 0 is <pad> in vocab
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1, seq_len, 1)
    return mask


if __name__ == "__main__":
    tgt_vocab_size = 10000
    max_len = 12
    pad_idx = 0
    d_model = 512
    N = 6
    d_ff = 2048
    heads_num = 8
    dropout_rate = 0.1
    layer_norm_eps = 1e-5

    batch_size = 5

    x = torch.randn(batch_size, max_len, d_model).gt(0)
    y = torch.randn(batch_size, max_len, d_model).gt(0)

    decoder = TransformerDecoder(
        tgt_vocab_size,
        max_len,
        pad_idx,
        d_model,
        N,
        d_ff,
        heads_num,
        dropout_rate,
        layer_norm_eps,
    )

    output = torch.zeros((batch_size, max_len), dtype=torch.long)  # "<BOS>"
    output[:, 0] = 1  # "<BOS>"
    print(output)
    src = torch.randn(batch_size, max_len, d_model)
    mask_src_tgt = torch.randn(batch_size, max_len, max_len).gt(0.5)
    mask_self = torch.randn(batch_size, max_len, max_len).gt(0.5)

    linear = nn.Linear(d_model, tgt_vocab_size)

    for t in range(max_len - 1):
        # mask_tgt = subsequent_mask(output)
        out = decoder(output, src, mask_src_tgt, mask_self)
        out = linear(out)
        out = torch.argmax(out, dim=2)
        output[:, t + 1] = out[:, t + 1]
        print(output)
