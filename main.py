from os.path import join

from const.path import TOK_CORPUS_PATH
from utils.vocab import get_vocab, tokenize_sentence

if __name__ == "__main__":
    vocab = get_vocab(join(TOK_CORPUS_PATH, "kyoto-train.en"))
    # print(vocab["<s>"])
    print(len(vocab))
    print(
        vocab.forward(
            tokenize_sentence(
                "<unk> He He is reputed to have been the one that spread the practices of tooth brushing , face washing , table manners and cleaning in Japan . <eos>"
            ),
            ["<pad>", "<unk>", "<eos>", "<bos>"],
        )
    )
