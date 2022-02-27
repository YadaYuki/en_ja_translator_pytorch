import torch

from layers.transformer.MultiHeadAttention import MultiHeadAttention


def test_sample() -> None:
    d_model = 512
    max_len = 128
    n_heads = 8
    attn = MultiHeadAttention()
    assert torch.equal(
        attn(torch.zeros(1, d_model, max_len)), torch.zeros(1, d_model, max_len)
    )
