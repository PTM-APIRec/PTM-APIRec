from transformers import (PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig,
                          T5ForConditionalGeneration)
from datasets import load_dataset


def load_model_dataset(file: str,
                       tokenizer: PreTrainedTokenizer,
                       batch_size: int,
                       model_type: str = None,
                       max_token_len: int = 1024,
                       test: bool = False,
                       encode_fn: callable = None,
                       file_type: str = "json",
                       api_pad_id: int = 0):
    if not test:
        train_ds = load_dataset(file_type, data_files=file, split="train[:90%]")
        valid_ds = load_dataset(file_type, data_files=file, split="train[90%:]")
        train_ds.shuffle()
        valid_ds.shuffle()
    else:
        train_ds = load_dataset(file_type, data_files=file, split="train")

    if encode_fn is None:
        def _default_encode_fn(samples):
            code_str = samples["code_str"]
            if model_type == "t5":
                code_str = [cs.split() for cs in code_str]
                for cs in code_str:
                    cs[-1] = "<extra_id_0>"
                code_str = [" ".join(cs) for cs in code_str]

            item = tokenizer(code_str,
                             padding="max_length",
                             max_length=max_token_len)

            candidates_batch = samples["candidates"]
            max_len = max(max(map(len, candidates_batch)), 10)
            candidates = []
            for cand in candidates_batch:
                cand += [api_pad_id] * (max_len - len(cand))
                candidates.append(cand)
            item["candidates"] = candidates

            item["target"] = samples["target"]

            if model_type == "t5":
                batch_len = len(samples["target"])
                labels = ["<extra_id_0>"] * batch_len
                labels = tokenizer(labels).input_ids
                item["labels"] = labels

            if test:
                tmp_ans = []
                tmp_tar = item["target"]
                tmp_cand = item["candidates"]
                for cand, tar in zip(tmp_cand, tmp_tar):
                    tmp_ans.append(cand[tar])
                item["ans"] = tmp_ans

            return item

        encode_fn = _default_encode_fn

    train_ds = train_ds.map(encode_fn, batched=True, batch_size=batch_size)
    if not test:
        valid_ds = valid_ds.map(encode_fn, batched=True, batch_size=batch_size)

    columns = ["input_ids", "attention_mask", "candidates", "target"]

    if model_type == "t5":
        columns.append("labels")
    if test:
        columns.append("ans")

    train_ds.set_format(type="torch", columns=columns)
    if not test:
        valid_ds.set_format(type="torch", columns=columns)

    return train_ds if test else (train_ds, valid_ds)


def load_pre_trained_tokenizer(file: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(file)
    return tokenizer


def load_pre_trained_model(file: str, config_only: bool = False, model_type: str = None) -> PreTrainedModel:
    _model_class = AutoModel
    if model_type == "t5":
        _model_class = T5ForConditionalGeneration
    if config_only:
        config = AutoConfig.from_pretrained(file, local_files_only=True)
        model = _model_class.from_config(config)
    else:
        model = _model_class.from_pretrained(file, local_files_only=True)
    return model


def load_wo_data(file: str,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,
                 max_token_len: int = 1024,
                 test: bool = False,
                 encode_fn: callable = None,
                 file_type: str = "json",
                 api_pad_id: int = 0):
    if not test:
        train_ds = load_dataset(file_type, data_files=file, split="train[:90%]")
        valid_ds = load_dataset(file_type, data_files=file, split="train[90%:]")
    else:
        train_ds = load_dataset(file_type, data_files=file, split="train")

    train_ds.shuffle()
    if not test:
        valid_ds.shuffle()

    if encode_fn is None:
        def _default_encode_fn(samples):
            item = tokenizer(samples["code_str"],
                             padding="max_length",
                             max_length=max_token_len)
            candidates_batch = samples["candidates"]
            target_batch = samples["target"]
            answer_batch = []
            for cand, tar in zip(candidates_batch, target_batch):
                answer_batch.append(cand[tar])
            item["target"] = answer_batch
            return item

        encode_fn = _default_encode_fn

    train_ds = train_ds.map(encode_fn, batched=True, batch_size=batch_size)
    if not test:
        valid_ds = valid_ds.map(encode_fn, batched=True, batch_size=batch_size)
        train_ds.set_format(type="torch", columns=["input_ids",
                                                   "attention_mask",
                                                   "target"])

        valid_ds.set_format(type="torch", columns=["input_ids",
                                                   "attention_mask",
                                                   "target"])
        return train_ds, valid_ds
    else:
        train_ds.set_format(type="torch", columns=["input_ids",
                                                   "attention_mask",
                                                   "target"])
        return train_ds
