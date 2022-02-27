from os.path import join

import torch
from const.path import TOK_CORPUS_PATH

from utils.text.text import tensor_to_text, text_to_tensor
from utils.text.vocab import BOS, EOS, UNK, get_vocab


def test_text_to_tensor() -> None:
    vocab = get_vocab(join(TOK_CORPUS_PATH, "kyoto-train.en"))
    # test empty text
    assert torch.equal(
        text_to_tensor("", vocab, 100, False, False), torch.zeros(100).to(torch.long)
    )

    # test "his hoge"
    his_hogehoge = "His hogehoge"
    expected = torch.zeros(100)
    expected[0] = vocab[BOS]
    expected[1] = vocab["His"]
    expected[2] = vocab[UNK]
    expected[3] = vocab[EOS]
    assert torch.equal(
        text_to_tensor(his_hogehoge, vocab, 100), expected.to(torch.long)
    )


def test_tensor_to_text() -> None:
    vocab = get_vocab(join(TOK_CORPUS_PATH, "kyoto-train.en"))

    # test "his hoge"
    assert tensor_to_text(text_to_tensor("His hogehoge", vocab, 4), vocab).startswith(
        f"{BOS} His {UNK} {EOS}"
    )
