# from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from const.path import TOK_CORPUS_PATH
from layers.transformer.Embedding import Embedding

if __name__ == "__main__":

    # en_train_path = join(join(TOK_CORPUS_PATH, "kyoto-train.en"))
    # vocab = get_vocab(en_train_path)
    # text_to_ids = lambda text, max_len: text_to_tensor(text, vocab, max_len)

    # d_model = 512
    # max_len = 128
    # vocab_size = len(vocab)

    # train_ds = KfttDataset(en_train_path, max_len, text_to_ids)
    # train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    # emb = Embedding(vocab_size, d_model, vocab[PAD])
    # pe = AddPositionalEncoding(d_model, max_len)

    # for batch in train_dl:
    #     emb_output = emb(batch)
    #     print(emb_output.shape)
    #     pe_output = pe(emb_output)
    #     print(pe_output.shape)
    #     break

    batch_size = 2
    d_k = 5
    t_l = 4
    x = torch.rand(1, batch_size * d_k * t_l).reshape(batch_size, t_l, d_k)
    # print(x.shape, x)
    # i次元目とj次元目を転置する.
    # まぁ、この辺の実装は慣れだなぁ
    x_t = torch.transpose(x, 1, 2)
    scalar = np.sqrt(d_k)
    # print(torch.matmul(x[0], x_t[0]))
    attention_weight = torch.nn.functional.softmax(torch.matmul(x, x_t) / scalar, dim=2)
    print(torch.matmul(attention_weight, x))
    attn = ScaledDotProductAttention(d_k)
    print(attn(x, x, x))
