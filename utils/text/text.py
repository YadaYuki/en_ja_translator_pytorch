import torch
from torchtext.vocab import Vocab
from utils.text.vocab import BOS, EOS, PAD, UNK, tokenize_sentence


def text_to_tensor(
    text: str, vocab: Vocab, max_len: int, eos: bool = True, bos: bool = True
) -> torch.Tensor:
    tokenized_text = tokenize_sentence(text)
    if eos:
        tokenized_text = tokenized_text + [EOS]
    if bos:
        tokenized_text = [BOS] + tokenized_text

    tensor = torch.zeros(max_len)
    for i in range(max_len):
        if i < len(tokenized_text):
            if tokenized_text[i] in vocab:
                tensor[i] = vocab[tokenized_text[i]]
            else:
                tensor[i] = vocab[UNK]
        else:
            tensor[i] = vocab[PAD]
    return tensor.to(torch.long)


def tensor_to_text(tensor: torch.Tensor, vocab: Vocab) -> str:
    text = []
    for i in range(tensor.size(0)):
        text.append(vocab.lookup_token(tensor[i]))
    return " ".join(text)
