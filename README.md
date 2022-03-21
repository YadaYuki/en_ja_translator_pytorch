# English to Japanese Translator by [pytorch](https://pytorch.org/) 🙊 ([Transformer](https://arxiv.org/abs/1706.03762) from scratch)

[![GitHub license](https://img.shields.io/github/license/YadaYuki/en_ja_translator_pytorch)](https://github.com/YadaYuki/en_ja_translator_pytorch) [![GitHub issues](https://img.shields.io/github/issues/YadaYuki/en_ja_translator_pytorch)](https://github.com/YadaYuki/en_ja_translator_pytorch/issues) [![GitHub forks](https://img.shields.io/github/forks/YadaYuki/en_ja_translator_pytorch)](https://github.com/YadaYuki/en_ja_translator_pytorch/network) [![GitHub stars](https://img.shields.io/github/stars/YadaYuki/en_ja_translator_pytorch)](https://github.com/YadaYuki/en_ja_translator_pytorch/stargazers)

## Overview

- English to Japanese translator by [Pytorch](https://pytorch.org/).
- The neural network architecture is [Transformer](https://arxiv.org/abs/1706.03762).
- The layers for Transfomer are implemented from scratch by pytorch. (you can find them under [layers/transformer/](https://github.com/YadaYuki/en_ja_translator_pytorch/tree/master/layers/transformer))
- Parallel corpus(dataset) is [kftt](http://www.phontron.com/kftt/index-ja.html).

## Transformer

![image](https://user-images.githubusercontent.com/57289763/159227403-edf771bf-e639-48f7-befe-763471e646da.png)

- Transformer is a neural network model proposed in the paper ‘[Attention Is All You Need](https://arxiv.org/abs/1706.03762)’

- As the paper's title said, transformer is a model based on Attention mechanism. Transformer does not use recursive calculation when training like RNN,LSTM
- Many of the models that have achieved high accuracy in various tasks in the NLP domain in recent years, such as BERT, GPT-3, and XLNet, have a Transformer-based structure.

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

set PYTHONPATH

```
export PYTHONPATH="$(pwd)"
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

You can train model by running:

```
$ poetry run python train.py

epoch: 1
--------------------Train--------------------

train loss: 10.104473114013672, bleu score: 0.0,iter: 1/4403

train loss: 9.551202774047852, bleu score: 0.0,iter: 2/4403

train loss: 8.950608253479004, bleu score: 0.0,iter: 3/4403

train loss: 8.688143730163574, bleu score: 0.0,iter: 4/4403

train loss: 8.4220552444458, bleu score: 0.0,iter: 5/4403

train loss: 8.243291854858398, bleu score: 0.0,iter: 6/4403

train loss: 8.187620162963867, bleu score: 0.0,iter: 7/4403

train loss: 7.6360859870910645, bleu score: 0.0,iter: 8/4403

....
```

- For each epoch, the model at that point is saved under pickles/nn/
- When the training is finished, loss.png is saved under figure/

## Reference

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Licence

[MIT](https://github.com/YadaYuki/en_ja_translator_pytorch/blob/master/LICENSE)
