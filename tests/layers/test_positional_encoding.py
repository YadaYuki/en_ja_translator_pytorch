import torch

from layers.transformer.PositionalEncoding import AddPositionalEncoding


def test_pe_shape() -> None:
    d_model = 512
    max_len = 128
    batch_size = 5
    pe = AddPositionalEncoding(d_model=d_model, max_len=max_len)
    output = pe(torch.randn(batch_size, max_len, d_model))
    assert output.shape == (batch_size, max_len, d_model)
