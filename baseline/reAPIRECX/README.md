# re-APIRecX

my reimplementation of APIRecX

(ori: https://github.com/yuningkang/APIRecX)

## train
To build a subword language model, first you need to subword your raw API sequence data with bpe.
(subword split with white space and each API ends with a <\t>)

Then generate tokenizer (model & vocab) using the subworded data with "word" tokenization mode in sentence-piece(here).

Use subworded data and above model/vocab to pretrain.

`python main.py`

## test

`python test.py`