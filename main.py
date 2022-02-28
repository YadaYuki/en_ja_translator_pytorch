# from os.path import join

import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from const.path import TOK_CORPUS_PATH
from layers.transformer.ScaledDotProductAttention import ScaledDotProductAttention

# importlib.reload(layers.transformer.ScaledDotProductAttention)
"""
6.3.5.1 Scaled Dot-Product Attention - PyTorch
"""

import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttentionAns(nn.Module):
    def __init__(self, d_k, device="cpu"):
        super().__init__()
        self.device = device
        self.scaler = np.sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        """
        # Argument
            q, k, v: (batch, sequence, out_features)
            mask:    (batch, sequence)
        """
        score = torch.einsum("ijk,ilk->ijl", (q, k)) / self.scaler
        score = score - torch.max(score, dim=-1, keepdim=True)[0]

        score = torch.exp(score)
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(1).repeat(1, score.size(1), 1)
            score.data.masked_fill_(mask, 0)
        a = score / torch.sum(score, dim=-1, keepdim=True)
        c = torch.einsum("ijk,ikl->ijl", (a, v))

        return c


def pad_mask(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    mask = x.eq(0)  # 0 is <pad> in vocab
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1, seq_len, 1)
    return mask


if __name__ == "__main__":

    # en_train_path = join(join(TOK_CORPUS_PATH, "kyoto-train.en"))
    # vocab = get_vocab(en_train_path)
    # text_to_ids = lambda text, max_len: text_to_tensor(text, vocab, max_len)

    # d_model = 512
    # max_len = 128
    # vocab_size = len(vocab)

    # train_ds = KfttDataset(en_train_path, max_len, text_to_ids)
    # train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    # emb = Embedding(vocab_size, d_model, vocab[PAD])
    # pe = AddPositionalEncoding(d_model, max_len)

    # for batch in train_dl:
    #     emb_output = emb(batch)
    #     print(emb_output.shape)
    #     pe_output = pe(emb_output)
    #     print(pe_output.shape)
    #     break

    batch_size = 2
    d_k = 5
    t_l = 4
    x = torch.rand(1, batch_size * d_k * t_l).reshape(batch_size, t_l, d_k)
    # print(x.shape, x)
    # i次元目とj次元目を転置する.
    # まぁ、この辺の実装は慣れだなぁ
    x_t = torch.transpose(x, 1, 2)
    scalar = np.sqrt(d_k)
    # print(torch.matmul(x[0], x_t[0]))

    attention_weight = torch.nn.functional.softmax(torch.matmul(x, x_t) / scalar, dim=2)
    # print(torch.matmul(attention_weight, x))
    attn = ScaledDotProductAttention(d_k)
    attn_ans = ScaledDotProductAttentionAns(d_k)
    m = pad_mask(
        torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 2, 0],
            ],
        ),
        t_l,
    )
    # print(m)
    mm = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 2, 0],
        ]
    ).eq(0)
    # print(attn(x, x, x))
    # print(attn_ans(x, x, x))
    print(attn(x, x, x, m))
    print(attn_ans(x, x, x, mm))
