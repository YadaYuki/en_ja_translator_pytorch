from os.path import join

import torch
from const.path import TOK_CORPUS_PATH

from utils.text import text_to_tensor
from utils.vocab import BOS, EOS, UNK, get_vocab


def test_text_to_tensor() -> None:
    vocab = get_vocab(join(TOK_CORPUS_PATH, "kyoto-train.en"))
    # test empty text
    assert torch.equal(text_to_tensor("", vocab, 100, False, False), torch.zeros(100))

    # test "his hoge"
    his_hogehoge = "His hogehoge"
    expected = torch.zeros(100)
    expected[0] = vocab[BOS]
    expected[1] = vocab["His"]
    expected[2] = vocab[UNK]
    expected[3] = vocab[EOS]
    assert torch.equal(text_to_tensor(his_hogehoge, vocab, 100), expected)
