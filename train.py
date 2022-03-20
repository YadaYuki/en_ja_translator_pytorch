from os.path import join
from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from const.path import (
    FIGURE_PATH,
    KFTT_TOK_CORPUS_PATH,
    NN_MODEL_PICKLES_PATH,
    TANAKA_CORPUS_PATH,
)
from models import Transformer

# from su.Transformer import Transformer as TransformerSugomori
from utils.dataset.Dataset import KfttDataset
from utils.evaluation.bleu import BleuScore
from utils.text.text import tensor_to_text, text_to_tensor
from utils.text.vocab import get_vocab


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        critetion: nn.Module,
        bleu_score: BleuScore,
        device: torch.device,
    ) -> None:
        self.net = net
        self.optimizer = optimizer
        self.critetion = critetion
        self.device = device
        self.bleu_score = bleu_score
        self.net = self.net.to(self.device)

    def loss_fn(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.critetion(preds, labels)

    def train_step(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        self.net.train()
        output = self.net(src, tgt)

        tgt = tgt[:, 1:]  # decoderからの出力は1 ~ max_lenまでなので、0以降のデータで誤差関数を計算する
        output = output[:, :-1, :]  #

        # calculate loss
        loss = self.loss_fn(
            output.contiguous().view(
                -1,
                output.size(-1),
            ),
            tgt.contiguous().view(-1),
        )

        # calculate bleu score
        _, output_ids = torch.max(output, dim=-1)
        bleu_score = self.bleu_score(tgt, output_ids)

        with torch.autograd.detect_anomaly():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, output, bleu_score

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

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, print_log: bool = True
    ) -> Tuple[List[float], List[float], List[float]]:
        # train
        train_losses: List[float] = []
        bleu_scores: List[float] = []
        if print_log:
            print(f"{'-'*20 + 'Train' + '-'*20}")
        for i, (src, tgt) in enumerate(train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _, bleu_score = self.train_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            if print_log:
                print(
                    f"train loss: {loss.item()}, bleu score: {bleu_score}"
                    + f"iter: {i+1}/{len(train_loader)}"
                )

            train_losses.append(loss.item())
            bleu_scores.append(bleu_score)

        # validation
        val_losses: List[float] = []
        if print_log:
            print(f"{'-'*20 + 'Validation' + '-'*20}")
        for i, (src, tgt) in enumerate(val_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _ = self.val_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            if print_log:
                print(f"train loss: {loss.item()}, iter: {i+1}/{len(val_loader)}")

            val_losses.append(loss.item())

        return train_losses, bleu_scores, val_losses

    def test(self, test_data_loader: DataLoader) -> List[float]:
        test_losses: List[float] = []
        for i, (src, tgt) in enumerate(test_data_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            loss, _ = trainer.val_step(src, tgt)
            src = src.to("cpu")
            tgt = tgt.to("cpu")

            test_losses.append(loss.item())

        return test_losses


if __name__ == "__main__":

    """
    1.define path & create vocab
    """
    TRAIN_SRC_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-train.en")
    TRAIN_TGT_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-train.ja")

    VAL_SRC_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-dev.en")
    VAL_TGT_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-dev.ja")

    TEST_SRC_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-test.en")
    TEST_TGT_CORPUS_PATH = join(KFTT_TOK_CORPUS_PATH, "kyoto-test.ja")

    src_vocab = get_vocab(TRAIN_SRC_CORPUS_PATH, vocab_size=20000)
    tgt_vocab = get_vocab(TRAIN_TGT_CORPUS_PATH, vocab_size=20000)

    """
    2.Define Parameters # TODO: from arguement or config file(hydra)
    """
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    max_len = 24
    d_model = 128
    heads_num = 4
    d_ff = 256
    N = 3
    dropout_rate = 0.1
    layer_norm_eps = 1e-8
    pad_idx = 0
    batch_size = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 2

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

    TEST_SRC_CORPUS_PATH
    TEST_TGT_CORPUS_PATH
    test_dataset = KfttDataset(
        TEST_SRC_CORPUS_PATH,
        TEST_TGT_CORPUS_PATH,
        max_len,
        src_text_to_tensor,
        tgt_text_to_tensor,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    """
    5.Train
    """
    trainer = Trainer(
        net,
        optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True),
        nn.CrossEntropyLoss(),
        BleuScore(tgt_vocab),
        device,
    )
    train_losses: List[float] = []
    val_losses: List[float] = []
    bleu_scores: List[float] = []
    for i in range(epoch):
        print(f"epoch: {i}")
        (
            train_losses_per_epoch,
            bleu_scores_per_epoch,
            val_losses_per_epoch,
        ) = trainer.fit(train_loader, val_loader, print_log=True)
        train_losses.extend(train_losses_per_epoch)
        bleu_scores.extend(bleu_scores_per_epoch)
        val_losses.extend(val_losses_per_epoch)
        torch.save(trainer.net, join(NN_MODEL_PICKLES_PATH, f"epoch_{i}.pt"))

    """
    6.Test
    """
    test_losses = trainer.test(test_loader)
    """
    7.Plot & save
    """
    plt.plot(list(range(len(train_losses))), train_losses, label="train")
    plt.plot(list(range(len(bleu_scores))), bleu_scores, label="bleu_score")
    plt.legend()
    plt.savefig(join(FIGURE_PATH, "train_loss.png"))

    print(f"train_losses: {train_losses}")
    print(f"val_losses: {val_losses}")
    print(f"test_losses: {test_losses}")
