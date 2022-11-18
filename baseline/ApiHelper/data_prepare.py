import torch
from torch.utils.data import Dataset

from tqdm import tqdm

from typing import Dict, List, Tuple, Union
import json

from data_service import download_method_item


PAD_STR = "<PAD>"
UNK_STR = "<UNK>"
HOLE_STR = "<HOLE>"

API_NAME_KEY = "apiName"
CLASS_NAME_KEY = "className"
TYPE_KEY = "type"


def build_api_dicts(api_vocab_file: str) -> (Dict[str, int], Dict[str, int], Dict[int, List[int]]):
    with open(api_vocab_file, "r") as fp:
        api_vocab = json.load(fp)
    api_dict: Dict[str, int] = {PAD_STR: 0,
                                UNK_STR: 1,
                                HOLE_STR: 2}
    class_dict: Dict[str, int] = {PAD_STR: 0,
                                  HOLE_STR: 1}
    class_to_api_dict: Dict[int, List[int]] = {}
    current_api_id = len(api_dict)
    current_class_id = len(class_dict)
    for api_item in api_vocab:
        api_name = api_item[API_NAME_KEY]
        class_name = api_item[CLASS_NAME_KEY]
        api_type = api_item[TYPE_KEY]
        if class_name not in class_dict:
            class_dict[class_name] = current_class_id
            class_to_api_dict[current_class_id] = []
            current_class_id += 1
        if api_name not in api_dict:
            api_dict[api_name] = current_api_id
            current_api_id += 1
        class_id = class_dict[class_name]
        api_id = api_dict[api_name]
        if api_id not in class_to_api_dict[class_id]:
            class_to_api_dict[class_id].append(api_id)
    return api_dict, class_dict, class_to_api_dict


def build_dataset(api_vocab_file: str,
                  method_id_file: str,
                  save_path: Union[str, Tuple[str, str]],
                  split_type: bool = False,
                  seq_max_len: int = 128):
    api_dict, class_dict, class_to_api_dict = build_api_dicts(api_vocab_file)
    with open(method_id_file, "r") as fp:
        method_ids = json.load(fp)
    jdk_samples = []
    android_samples = []
    for mid in tqdm(method_ids):
        method_item = download_method_item(mid)
        api_list = method_item["api_seq"]
        if len(api_list) < 2:
            continue
        for i in range(1, len(api_list)):
            before_hole_apis = api_list[:i]
            api_ids = [api_dict.get(x[API_NAME_KEY], UNK_STR) for x in before_hole_apis]
            class_ids = [class_dict.get(x[CLASS_NAME_KEY], UNK_STR) for x in before_hole_apis]
            hole_api = api_list[i]
            gt_api_id = api_dict.get(hole_api[API_NAME_KEY], UNK_STR)
            gt_class_id = class_dict.get(hole_api[CLASS_NAME_KEY], UNK_STR)
            api_ids.append(api_dict[HOLE_STR])
            class_ids.append(gt_class_id)
            target = gt_api_id
            target_class = gt_class_id

            seq_len = len(api_ids)
            if seq_len > seq_max_len:
                api_ids = api_ids[-seq_max_len:]
                class_ids = class_ids[-seq_max_len:]

            sample = {
                "api_ids": api_ids,
                "class_ids": class_ids,
                "target": target,
                "target_class": target_class
            }
            if (not split_type) or (hole_api[TYPE_KEY] == "standard"):
                jdk_samples.append(sample)
            else:
                android_samples.append(sample)
    if not split_type:
        with open(save_path, "w+") as fp:
            json.dump(jdk_samples, fp=fp)
    else:
        with open(save_path[0], "w+") as fp:
            json.dump(jdk_samples, fp=fp)
        with open(save_path[1], "w+") as fp:
            json.dump(android_samples, fp=fp)

