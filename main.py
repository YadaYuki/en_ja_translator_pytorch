from os.path import join

import torch
from torch.utils.data import DataLoader

from const.path import TOK_CORPUS_PATH
from layers.transformer.embedding import Embedding
from utils.dataset.Dataset import KfttDataset
from utils.text.text import tensor_to_text, text_to_tensor
from utils.text.vocab import PAD, get_vocab

if __name__ == "__main__":

    en_train_path = join(join(TOK_CORPUS_PATH, "kyoto-train.en"))
    vocab = get_vocab(en_train_path)
    text_to_ids = lambda text, max_len: text_to_tensor(text, vocab, max_len)
    train_ds = KfttDataset(en_train_path, 128, text_to_ids)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    emb_dim = 128
    emb = Embedding(len(vocab), emb_dim, vocab[PAD])
    for batch in train_dl:
        print(emb(batch))
        break
