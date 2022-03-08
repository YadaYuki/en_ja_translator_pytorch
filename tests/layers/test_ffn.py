import torch

from layers.transformer.FFN import FFN


def test_ffn_shape() -> None:
    d_model = 512
    d_ff = 2048
    max_len = 128
    batch_size = 5
    ffn = FFN(d_model=d_model, d_ff=d_ff)
    output = ffn(torch.randn(batch_size, max_len, d_model))
    assert output.shape == (batch_size, max_len, d_model)
