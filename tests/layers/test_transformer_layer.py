import torch

from layers.transformer.FFN import FFN
from layers.transformer.PositionalEncoding import AddPositionalEncoding
from layers.transformer.TransformerEncoder import TransformerEncoderLayer


def test_ffn_shape() -> None:
    d_model = 512
    d_ff = 2048
    max_len = 128
    batch_size = 5
    ffn = FFN(d_model=d_model, d_ff=d_ff)
    output = ffn(torch.randn(batch_size, max_len, d_model))
    assert output.shape == (batch_size, max_len, d_model)


def test_pe_shape() -> None:
    d_model = 512
    max_len = 128
    batch_size = 5
    pe = AddPositionalEncoding(d_model=d_model, max_len=max_len)
    output = pe(torch.randn(batch_size, max_len, d_model))
    assert output.shape == (batch_size, max_len, d_model)


def test_transformer_encoder_layer_shape() -> None:
    d_model = 512
    d_ff = 2048
    heads_num = 8
    dropout_rate = 0.1
    layer_norm_eps = 1e-6
    max_len = 128
    batch_size = 5
    transformer_encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        d_ff=d_ff,
        heads_num=heads_num,
        dropout_rate=dropout_rate,
        layer_norm_eps=layer_norm_eps,
    )
    x = torch.randn(batch_size, max_len, d_model)
    mask = torch.randn(batch_size, max_len, max_len).eq(0)
    output = transformer_encoder_layer(x, mask)
    assert output.shape == (batch_size, max_len, d_model)
