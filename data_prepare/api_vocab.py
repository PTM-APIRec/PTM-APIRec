from __future__ import annotations

import json
from collections import defaultdict
from typing import List


class ApiVocabulary:
    def __init__(self,
                 api_list: List[dict],
                 special_tokens: List[str] = None,
                 require_signature: bool = False):
        self.require_signature = require_signature
        self.api_to_id = {}
        self.class_api = defaultdict(list)
        if special_tokens is None:
            self.pad_token = "<PAD>"
            self.unk_token = "<UNK>"
            special_tokens = [
                self.pad_token,
                self.unk_token
            ]
        api_id = 0
        for sp_token in special_tokens:
            self.api_to_id[sp_token] = api_id
            api_id += 1
        for api in api_list:
            if not self.require_signature:
                api_name = api["class_name"] + "." + api["api_name"]
            else:
                api_name = api["signature"]
            if api_name not in self.api_to_id:
                self.api_to_id[api_name] = api_id
                class_name = api["class_name"]
                self.class_api[class_name].append(api_id)
                api_id += 1
        self.id_to_api = {self.api_to_id[k]: k for k in self.api_to_id.keys()}

    def __getitem__(self, api_name: str) -> int:
        return self.get_api_id(api_name)

    def __len__(self) -> int:
        return len(self.api_to_id)

    def __contains__(self, api_name: str) -> bool:
        return api_name in self.api_to_id

    def get_class_api(self, class_name: str) -> List[id]:
        return self.class_api.get(class_name, [])

    def get_api_id(self, api_name: str) -> int:
        return self.api_to_id.get(api_name, -1)

    def get_api_name(self, api_id: int) -> str:
        return self.id_to_api[api_id]

    @staticmethod
    def from_json(file: str, filter_fn: callable = None) -> ApiVocabulary:
        with open(file, "r") as fp:
            raw_api_list = json.load(fp)
        api_list = []
        if filter_fn is None:
            filter_fn = _default_filter_fn
        raw_api_list = list(filter(filter_fn, raw_api_list))
        for api in raw_api_list:
            item = {
                "api_name": api["apiName"],
                "class_name": api["className"]
            }
            api_list.append(item)
        return ApiVocabulary(api_list)


def _default_filter_fn(api: dict) -> bool:
    return api["apiName"] != "init" and api["className"] != "CONTROL"
