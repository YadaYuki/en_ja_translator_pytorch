# English to Japanese Translator by [pytorch](https://pytorch.org/) 🙊 ([Transformer](https://arxiv.org/abs/1706.03762) from scratch)

[![GitHub license](https://img.shields.io/github/license/YadaYuki/en_ja_translator_pytorch)](https://github.com/YadaYuki/en_ja_translator_pytorch) [![GitHub issues](https://img.shields.io/github/issues/YadaYuki/en_ja_translator_pytorch)](https://github.com/YadaYuki/en_ja_translator_pytorch/issues) [![GitHub forks](https://img.shields.io/github/forks/YadaYuki/en_ja_translator_pytorch)](https://github.com/YadaYuki/en_ja_translator_pytorch/network) [![GitHub stars](https://img.shields.io/github/stars/YadaYuki/en_ja_translator_pytorch)](https://github.com/YadaYuki/en_ja_translator_pytorch/stargazers)

## Overview

- English to Japanese translator by [Pytorch](https://pytorch.org/).
- The neural network architecture is [Transformer](https://arxiv.org/abs/1706.03762).
- The layers for Transfomer are implemented from scratch by pytorch. (you can find them under [layers/transformer/](https://github.com/YadaYuki/en_ja_translator_pytorch/tree/master/layers/transformer))
- Parallel corpus(dataset) is [kftt](http://www.phontron.com/kftt/index-ja.html).

## Transformer

## Requirements

- [poetry](https://python-poetry.org/) 1.0.10+
- [python](https://www.python.org/) 3.8+

## Setup

Install dependencies & create a virtual environment in project by running:

```
$ poetry install
```

Download & unzip parallel corpus([kftt](http://www.phontron.com/kftt/index-ja.html)) by running:

```
$ poetry run python ./utils/download.py
```

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

## How to run

## Reference
