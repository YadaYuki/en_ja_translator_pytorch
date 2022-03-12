from os.path import join
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from const.path import TOK_CORPUS_PATH
from models import Transformer
from utils.dataset.Dataset import KfttDataset
from utils.text.text import text_to_tensor
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
        output = self.net(src, tgt)
        vocab_size = output.shape[-1]

        # convert tgt to one-hot
        tgt = F.one_hot(tgt, vocab_size).float()
        loss = self.loss_fn(tgt, output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, output

    def val_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.net(src, tgt)
        vocab_size = output.shape[-1]

        # convert tgt to one-hot
        tgt = F.one_hot(tgt, vocab_size).float()
        loss = self.loss_fn(tgt, output)

        return loss, output


if __name__ == "__main__":

    """
    1.load & create vocab
    """
    TRAIN_SRC_CORPUS_PATH = join(TOK_CORPUS_PATH, "kyoto-train.en")
    TRAIN_TGT_CORPUS_PATH = join(TOK_CORPUS_PATH, "kyoto-train.ja")

    src_vocab = get_vocab(TRAIN_SRC_CORPUS_PATH)
    tgt_vocab = get_vocab(TRAIN_TGT_CORPUS_PATH)

    """
    2.Define Parameters # TODO: from arguement or config file(hydra)
    """
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    max_len = 24
    d_model = 512
    heads_num = 8
    d_ff = 2048
    N = 6
    dropout_rate = 0.1
    layer_norm_eps = 1e-5
    pad_idx = 0
    batch_size = 128

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
    )

    """
    4.Define dataset & dataloader
    """

    train_dataset = KfttDataset(
        TRAIN_SRC_CORPUS_PATH,
        TRAIN_TGT_CORPUS_PATH,
        max_len,
        lambda text, max_len: text_to_tensor(text, src_vocab, max_len),
        lambda text, max_len: text_to_tensor(text, tgt_vocab, max_len),
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    """
    5.Train
    """
    trainer = Trainer(
        net,
        optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True),
        nn.CrossEntropyLoss(),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    for i, (src, tgt) in enumerate(train_loader):
        loss, output = trainer.train_step(src, tgt)
        print(f"loss: {loss.item()}, iter: {i}")
