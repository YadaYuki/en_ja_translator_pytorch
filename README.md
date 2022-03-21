# English to Japanese Translator by [pytorch](https://pytorch.org/) 🙊 ([Transformer](https://arxiv.org/abs/1706.03762) from scratch)

## Overview

- English to Japanese translator by Pytorch.
- The neural network architecture is [Transformer](https://arxiv.org/abs/1706.03762).
- The layers for Transfomer are implemented from scratch by pytorch. (you can find them under [layers/transformer/](https://github.com/YadaYuki/en_ja_translator_pytorch/tree/master/layers/transformer))
- Parallel corpus(dataset) is [kftt](http://www.phontron.com/kftt/index-ja.html).

## Transformer

## Directories

The directory structure is as below.

```
.
├── const
│   └── path.py
├── corpus
│   └── kftt-data-1.0
├── figure
├── layers
│   └── transformer
│       ├── Embedding.py
│       ├── FFN.py
│       ├── MultiHeadAttention.py
│       ├── PositionalEncoding.py
│       ├── ScaledDotProductAttention.py
│       ├── TransformerDecoder.py
│       └── TransformerEncoder.py
├── models
│   ├── Transformer.py
│   └── __init__.py
├── mypy.ini
├── pickles
│   └── nn/
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── tests
│   ├── conftest.py
│   ├── layers/
│   ├── models/
│   └── utils/
├── train.py
└── utils
    ├── dataset/
    ├── download.py
    ├── evaluation/
    └── text/
```

## Setup

## How to use

## Reference
