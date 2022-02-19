from os.path import join

from torch.utils.data import DataLoader

from const.path import TOK_CORPUS_PATH
from utils.dataset.Dataset import KfttDataset
from utils.text.text import tensor_to_text, text_to_tensor
from utils.text.vocab import get_vocab

if __name__ == "__main__":

    en_train_path = join(join(TOK_CORPUS_PATH, "kyoto-train.en"))
    vocab = get_vocab(en_train_path)
    # print(vocab["<s>"])
    text_to_ids = lambda text, max_len: text_to_tensor(text, vocab, max_len)
    train_ds = KfttDataset(en_train_path, 128, text_to_ids)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    for batch in train_dl:
        for i in range(len(batch)):
            print(tensor_to_text(batch[i], vocab))
        break
