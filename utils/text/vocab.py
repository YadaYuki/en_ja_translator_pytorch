from typing import Generator, List, Optional

from torchtext.vocab import Vocab, build_vocab_from_iterator

PAD = "<pad>"
UNK = "<unk>"
EOS = "<eos>"
BOS = "<bos>"


def get_vocab(
    path_to_corpus: str,
    specials: List[str] = [
        PAD,
        UNK,
        EOS,
        BOS,
    ],
    vocab_size: Optional[int] = None,
) -> Vocab:
    return build_vocab_from_iterator(_yield_token(path_to_corpus), specials=specials)


def _yield_token(path_to_corpus: str) -> Generator[List[str], None, None]:
    with open(path_to_corpus, "r", encoding="utf-8") as f:
        for line in f:
            yield tokenize_sentence(line)


def tokenize_sentence(sentence: str) -> List[str]:
    """トークンごとに空白で区切られた文章をトークンの配列に変換する。"""
    return sentence.strip().split()
