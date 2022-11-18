from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, _utils

import torch
import torch.nn as nn

import json
from typing import Union, List, Tuple


def load_model_dataset(file: str,
                       batch_size: int,
                       seq_max_len: int = 128,
                       test: bool = False,
                       encode_fn: callable = None,
                       file_type: str = "json",
                       pad_id: int = 0):
    if not test:
        train_ds = load_dataset(file_type, data_files=file, split="train[:90%]")
        valid_ds = load_dataset(file_type, data_files=file, split="train[90%:]")
        train_ds.shuffle()
        valid_ds.shuffle()
    else:
        train_ds = load_dataset(file_type, data_files=file, split="train")

    if encode_fn is None:

        def _default_encode_fn(samples):
            api_batch = samples["api_ids"]
            class_batch = samples["class_ids"]
            apis, classes = [], []
            for api_item, class_item in zip(api_batch, class_batch):
                seq_len = len(api_item)
                api_item += [0] * (seq_max_len - seq_len)
                class_item += [0] * (seq_max_len - seq_len)
                apis.append(api_item)
                classes.append(class_item)
            samples["api_ids"] = apis
            samples["class_ids"] = classes
            candidate_batch = samples["candidate_ids"]
            candidate_max_len = max(max(map(len, candidate_batch)), 10)
            candidates = []
            for cand in candidate_batch:
                cand += [pad_id] * (candidate_max_len - len(cand))
                candidates.append(cand)
            samples["candidate_ids"] = candidates
            return samples

        encode_fn = _default_encode_fn

    train_ds = train_ds.map(encode_fn, batched=True, batch_size=batch_size)
    if not test:
        valid_ds = valid_ds.map(encode_fn, batched=True, batch_size=batch_size)

    columns = ["api_ids", "class_ids", "candidate_ids", "target"]
    train_ds.set_format(type="torch", columns=columns)
    if not test:
        valid_ds.set_format(type="torch", columns=columns)

    return train_ds if test else (train_ds, valid_ds)


def load_model(model: nn.Module,
               path: str,
               map_location=None):
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.load(path, map_location=map_location)
    if type(states) == dict:
        model.load_state_dict(states["model"])
    else:
        model.load_state_dict(states)
    return model


class ApiHelperDataset(Dataset):
    def __init__(self,
                 api_dict: dict,
                 class_dict: dict,
                 class_to_api_dict: dict,
                 dataset: List[dict],
                 max_seq_len: int = 128,
                 pad_id: int = 0):
        self._api_dict = api_dict
        self._class_dict = class_dict
        self._class_to_api_dict = class_to_api_dict
        self._dataset = dataset
        self._max_seq_len = max_seq_len
        self._pad_id = pad_id
        self._api_vocab_size = len(self._api_dict)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        item = self._dataset[index]
        api_ids = item["api_ids"]
        class_ids = item["class_ids"]
        target = item["target"]
        target_class = item["target_class"]
        candidates = self._class_to_api_dict[target_class]
        api_ids += [self._pad_id] * (self._max_seq_len - len(api_ids))
        class_ids += [self._pad_id] * (self._max_seq_len - len(class_ids))

        api_ids = torch.LongTensor(api_ids)
        class_ids = torch.LongTensor(class_ids)
        candidate_ids = torch.zeros(self._api_vocab_size)
        candidate_ids[candidates] = 1

        return [((api_ids, class_ids, candidate_ids), target)]


def load_dataloader(file: str,
                    batch_size: int,
                    api_dict: dict,
                    class_dict: dict,
                    class_to_api_dict: dict,
                    max_seq_len: int = 128,
                    split: float = None,
                    pad_id: int = 0) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    with open(file, "r") as fp:
        samples = json.load(fp)
    if split is not None:
        total_size = len(samples)
        first_size = int(total_size * split)
        train_dataset = ApiHelperDataset(api_dict,
                                         class_dict,
                                         class_to_api_dict,
                                         dataset=samples[:first_size],
                                         max_seq_len=max_seq_len,
                                         pad_id=pad_id)
        valid_dataset = ApiHelperDataset(api_dict,
                                         class_dict,
                                         class_to_api_dict,
                                         dataset=samples[first_size:],
                                         max_seq_len=max_seq_len,
                                         pad_id=pad_id)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  collate_fn=_collate_fn,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  collate_fn=_collate_fn,
                                  shuffle=True)
        return train_loader, valid_loader
    else:
        test_dataset = ApiHelperDataset(api_dict,
                                        class_dict,
                                        class_to_api_dict,
                                        dataset=samples,
                                        max_seq_len=max_seq_len,
                                        pad_id=pad_id)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=_collate_fn,
                                 shuffle=False)
        return test_loader


def _collate_fn(batch_data):
    squeezed_batch_data = []
    for data_item in batch_data:
        for sample in data_item:
            squeezed_batch_data.append(sample)

    return _utils.collate.default_collate(squeezed_batch_data)
