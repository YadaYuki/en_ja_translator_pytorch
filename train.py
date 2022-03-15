from os.path import join
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from const.path import FIGURE_PATH, NN_MODEL_PICKLES_PATH, TANAKA_CORPUS_PATH
from models import Transformer
from su.Transformer import Transformer as TransformerSugomori
from utils.dataset.Dataset import KfttDataset
from utils.text.text import tensor_to_text, text_to_tensor
from utils.text.vocab import get_vocab


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        critetion: nn.Module,
        device: torch.device,
    ) -> None:
        self.net = net
        self.optimizer = optimizer
        self.critetion = critetion
        self.device = device
        self.net.to(self.device)

    def loss_fn(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.critetion(preds, labels)

    def train_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.net.train()
        output = self.net(src, tgt)
        # vocab_size = output.shape[-1]

        # convert tgt to one-hot
        tgt = tgt[:, 1:]  # decoderからの出力は1 ~ max_lenまでなので、0以降のデータで誤差関数を計算する
        output = output[:, :-1, :]  #
        # print(output.shape, tgt.shape)
        loss = self.loss_fn(
            output.contiguous().view(
                -1,
                output.size(-1),
            ),
            tgt.contiguous().view(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, output

    def val_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.net.eval()
        output = self.net(src, tgt)

        tgt = tgt[:, 1:]
        output = output[:, :-1, :]  #

        loss = self.loss_fn(
            output.contiguous().view(
                -1,
                output.size(-1),
            ),
            tgt.contiguous().view(-1),
        )

        return loss, output


if __name__ == "__main__":

    """
    1.define path & create vocab
    """
    # TRAIN_SRC_CORPUS_PATH = join(TOK_CORPUS_PATH, "kyoto-train.en")
    # TRAIN_TGT_CORPUS_PATH = join(TOK_CORPUS_PATH, "kyoto-train.ja")

    # VAL_SRC_CORPUS_PATH = join(TOK_CORPUS_PATH, "kyoto-dev.en")
    # VAL_TGT_CORPUS_PATH = join(TOK_CORPUS_PATH, "kyoto-dev.ja")

    # TEST_SRC_CORPUS_PATH = join(TOK_CORPUS_PATH, "kyoto-test.en")
    # TEST_TGT_CORPUS_PATH = join(TOK_CORPUS_PATH, "kyoto-test.ja")

    TRAIN_SRC_CORPUS_PATH = join(TANAKA_CORPUS_PATH, "train.en")
    TRAIN_TGT_CORPUS_PATH = join(TANAKA_CORPUS_PATH, "train.ja")

    VAL_SRC_CORPUS_PATH = join(TANAKA_CORPUS_PATH, "dev.en")
    VAL_TGT_CORPUS_PATH = join(TANAKA_CORPUS_PATH, "dev.ja")

    TEST_SRC_CORPUS_PATH = join(TANAKA_CORPUS_PATH, "test.en")
    TEST_TGT_CORPUS_PATH = join(TANAKA_CORPUS_PATH, "test.ja")

    src_vocab = get_vocab(TRAIN_SRC_CORPUS_PATH)
    tgt_vocab = get_vocab(TRAIN_TGT_CORPUS_PATH)

    """
    2.Define Parameters # TODO: from arguement or config file(hydra)
    """
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    # max_len = 24
    # d_model = 512
    # heads_num = 8
    # d_ff = 2048
    # N = 6
    # dropout_rate = 0.1
    # layer_norm_eps = 1e-5
    # pad_idx = 0
    # batch_size = 128
    max_len = 24
    d_model = 128
    heads_num = 4
    d_ff = 256
    N = 3
    dropout_rate = 0.1
    layer_norm_eps = 1e-8
    pad_idx = 0
    batch_size = 100

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    epoch = 10

    print(f"src_vocab_size: {src_vocab_size}")
    print(f"tgt_vocab_size: {tgt_vocab_size}")

    """
    3.Define model
    """
    net = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len,
        d_model=d_model,
        heads_num=heads_num,
        d_ff=d_ff,
        N=N,
        dropout_rate=dropout_rate,
        layer_norm_eps=layer_norm_eps,
        pad_idx=pad_idx,
        device=device,
    )
    # net = TransformerSugomori(
    #     src_vocab_size,
    #     tgt_vocab_size,
    #     N=3,
    #     h=4,
    #     d_model=128,
    #     d_ff=256,
    #     maxlen=max_len,
    #     device=device,
    # ).to(device)
    net.to(device)

    """
    4.Define dataset & dataloader
    """

    def src_text_to_tensor(text: str, max_len: int) -> torch.Tensor:
        return text_to_tensor(text, src_vocab, max_len, eos=False, bos=False)

    def src_tensor_to_text(tensor: torch.Tensor) -> str:
        return tensor_to_text(tensor, src_vocab)

    def tgt_text_to_tensor(text: str, max_len: int) -> torch.Tensor:
        return text_to_tensor(text, tgt_vocab, max_len)

    def tgt_tensor_to_text(tensor: torch.Tensor) -> str:
        return tensor_to_text(tensor, tgt_vocab)

    train_dataset = KfttDataset(
        TRAIN_SRC_CORPUS_PATH,
        TRAIN_TGT_CORPUS_PATH,
        max_len,
        src_text_to_tensor,
        tgt_text_to_tensor,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataset = KfttDataset(
        VAL_SRC_CORPUS_PATH,
        VAL_TGT_CORPUS_PATH,
        max_len,
        src_text_to_tensor,
        tgt_text_to_tensor,
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    """
    5.Train
    """
    trainer = Trainer(
        net,
        optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True),
        nn.CrossEntropyLoss(),
        device,
    )

    train_losses = []
    val_losses = []
    for i in range(epoch):
        print(f"epoch: {i}")

        # train
        print(f"{'-'*20 + 'train' + '-'*20}")
        train_loss = 0.0
        for i, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            loss, output = trainer.train_step(src, tgt)
            print()
            print(f"loss: {loss.item()}, iter: {i+1}/{len(train_loader)}")
            train_loss += loss.item()
            src = src.to("cpu")
            tgt = tgt.to("cpu")
            output_word_ids = output.max(-1)[1]
            for i in range(10):
                print(f"output: {tgt_tensor_to_text(output_word_ids[i])}")

        train_losses.append(train_loss / len(train_loader))

        # validation
        print(f"{'-'*20 + 'validation' + '-'*20}")
        val_loss = 0.0
        for i, (src, tgt) in enumerate(val_loader):
            loss, output = trainer.val_step(src.to(device), tgt.to(device))
            print(f"loss: {loss.item()}, iter: {i+1}/{len(val_loader)}")
            val_loss += loss.item()

            output_word_ids = output.max(-1)[1]
            for i in range(10):
                print(f"output: {tgt_tensor_to_text(output_word_ids[i])}")

        print(f"train_loss: {train_loss / len(train_loader)}")
        val_losses.append(val_loss / len(val_loader))

        # save model
        torch.save(net, join(NN_MODEL_PICKLES_PATH, f"epoch_{epoch}.pt"))

    """
    6.Test: FUTURE DEVELOPMENT!!! (TODO)
    """

    """
    7.Plot & save
    """
    x = list(range(epoch))
    plt.plot(x, train_loss, label="train")
    plt.plot(x, val_loss, label="val")
    plt.legend()
    plt.savefig(join(FIGURE_PATH, "loss.png"))
