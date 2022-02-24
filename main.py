from os.path import join

from torch.utils.data import DataLoader

from const.path import TOK_CORPUS_PATH
from layers.transformer.Embedding import Embedding
from layers.transformer.PositionalEncoding import AddPositionalEncoding
from utils.dataset.Dataset import KfttDataset
from utils.text.text import tensor_to_text, text_to_tensor
from utils.text.vocab import PAD, get_vocab

if __name__ == "__main__":

    en_train_path = join(join(TOK_CORPUS_PATH, "kyoto-train.en"))
    vocab = get_vocab(en_train_path)
    text_to_ids = lambda text, max_len: text_to_tensor(text, vocab, max_len)

    d_model = 512
    max_len = 128
    vocab_size = len(vocab)

    train_ds = KfttDataset(en_train_path, max_len, text_to_ids)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    emb = Embedding(vocab_size, d_model, vocab[PAD])
    pe = AddPositionalEncoding(d_model, max_len)
    for batch in train_dl:
        emb_output = emb(batch)
        print(emb_output.shape)
        pe_output = pe(emb_output)
        print(pe_output.shape)
        break
