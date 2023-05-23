from typing import Iterable, Union, List
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


class PretrainInputExample:
    """A single example for unsupervised pre-training.
    """

    def __init__(self, text: str):
        self.text = text


class PretrainInputFeatures:
    """A single set of features of pre-training data.
    """

    def __init__(self, input_ids: List[int]):
        self.input_ids = input_ids


def convert_examples_to_features(examples,
                                 tokenizer,
                                 args, ):
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token

    # Create features
    features = []
    for i, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = [bos_token] + tokens[:args.max_seq_len - 2] + [eos_token]  # BOS, EOS
        tokens += [pad_token] * (args.max_seq_len - len(tokens))
        # features_seq.append(" ".join(tokens))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        feature = PretrainInputFeatures(input_ids)
        features.append(feature)

    return features


def create_examples(args, tokenizer, mode='train'):
    # Load data features from cache or dataset file
    assert mode in ('train', 'valid', 'test')

    cached_features_file = Path('cached_features_{}_{}_{}_clean_bpe'.format('pretrain', mode, args.max_seq_len))
    if cached_features_file.exists():
        print('Loading features from cached file', cached_features_file)
        features = torch.load(cached_features_file)

    else:
        # corpus_path = args.train_corpus if mode == 'train' else args.valid_corpus
        if mode == 'train':
            corpus_path = args.train_corpus
        elif mode == 'valid':
            corpus_path = args.valid_corpus

        with open(corpus_path, 'r', encoding='utf-8') as reader:
            corpus = reader.readlines()

        # Create examples
        corpus = list(map(lambda x: x.strip(), corpus))
        corpus = list(filter(lambda x: len(x) > 0, corpus))
        examples = [PretrainInputExample(text) for text in corpus]

        # Convert examples to features
        features = convert_examples_to_features(examples, tokenizer, args)

        print('Saving features into cached file', cached_features_file)
        torch.save(features, cached_features_file)

    # Create dataset with features
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    # targets_ids = torch.tensor()
    print("mode {} all feature input ids shape: ".format(mode), all_input_ids.shape)
    # print("all feature input ids: ", all_input_ids)
    dataset = TensorDataset(all_input_ids)

    return dataset


def get_data_loaders(args, tokenizer):
    print("-------------create data loaders-------------")
    train_dataset = create_examples(args=args, tokenizer=tokenizer, mode='train')
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=args.n_workers)

    valid_dataset = create_examples(args=args, tokenizer=tokenizer, mode='valid')
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.n_workers)

    return train_loader, valid_loader
