import torch

from models.Transformer import Transformer


def test_transformer_train() -> None:
    src_vocab_size = 15000
    tgt_vocab_size = 10000
    max_len = 50
    batch_size = 10
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len,
    )

    # 訓練時
    src = torch.randint(src_vocab_size, (batch_size, max_len))
    tgt = torch.randint(tgt_vocab_size, (batch_size, max_len))

    output = transformer(src, tgt)

    assert output.shape == (batch_size, max_len, tgt_vocab_size)


def test_transformer_val() -> None:
    src_vocab_size = 15000
    tgt_vocab_size = 10000
    max_len = 50
    batch_size = 10
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len,
    )

    # 訓練時
    src = torch.randint(src_vocab_size, (batch_size, max_len))

    output = transformer(src)

    assert output.shape == (batch_size, max_len)
