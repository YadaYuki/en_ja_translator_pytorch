import torch
from layers.transformer.TransfomerDecoder import TransformerDecoder
from layers.transformer.TransformerEncoder import TransformerEncoder
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int,
        d_model: int = 512,
        heads_num: int = 8,
        d_ff: int = 2048,
        N: int = 6,
        dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-5,
        pad_idx: int = 0,
    ):

        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.heads_num = heads_num
        self.d_ff = d_ff
        self.N = N
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.pad_idx = pad_idx

        self.encoder = TransformerEncoder(
            src_vocab_size,
            max_len,
            pad_idx,
            d_model,
            N,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            max_len,
            pad_idx,
            d_model,
            N,
            d_ff,
            heads_num,
            dropout_rate,
            layer_norm_eps,
        )

        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        ----------
        src : torch.Tensor
            単語のid列. [batch_size, max_len]
        tgt : torch.Tensor
            単語のid列. [batch_size, max_len]
        """

        # mask
        pad_mask_src = self._pad_mask(src)

        src = self.encoder(src, pad_mask_src)

        if tgt is not None:

            # tgt = tgt[:, :-1]  # 入力は最後の<eos>を除く

            mask_self_attn = torch.logical_or(
                self._subsequent_mask(tgt), self._pad_mask(tgt)
            )

            dec_output = self.decoder(tgt, src, pad_mask_src, mask_self_attn)
            return self.linear(dec_output)
        else:
            batch_size = src.size(0)

            output = torch.ones((batch_size, self.max_len), dtype=torch.long)

            for t in range(self.max_len - 1):
                mask_tgt = self._subsequent_mask(output)
                out = self.decoder(output, src, pad_mask_src, mask_tgt)
                out = self.linear(out)
                out = torch.argmax(out, dim=-1)
                output[:, t + 1] = out

            return output

    def _pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """単語のid列(ex:[[4,1,9,11,0,0,0...],[4,1,9,11,0,0,0...],[4,1,9,11,0,0,0...]...])からmaskを作成する.
        Parameters:
        ----------
        x : torch.Tensor
            単語のid列. [batch_size, max_len]
        """
        seq_len = x.size(1)
        mask = x.eq(self.pad_idx)  # 0 is <pad> in vocab
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, seq_len, 1)  # (batch_size, max_len, max_len)
        return mask

    def _subsequent_mask(self, x: torch.Tensor) -> torch.Tensor:
        """DecoderのMasked-Attentionに使用するmaskを作成する.
        Parameters:
        ----------
        x : torch.Tensor
            単語のトークン列. [batch_size, max_len, d_model]
        """
        batch_size = x.size(0)
        max_len = x.size(1)
        return torch.tril(torch.ones(batch_size, max_len, max_len)).eq(0)
