from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer
from tqdm import tqdm

from data_prepare.mongo_service import *
from data_prepare.api_vocab import ApiVocabulary
from utils import find_all_locations

import json


def prepare_single_sample(method_id: str,
                          tokenizer: PreTrainedTokenizer,
                          api_vocab: ApiVocabulary,
                          max_token_len: int = 512,
                          target_api_type: str = "android",
                          mask_token: str = "<API_MASK>",
                          mlm: bool = False,
                          mlm_token: str = "<mask>",
                          mlm_mask_len: int = 5):
    method_item = download_method_item(method_id)
    masked_code = method_item["masked_src"]
    api_seq = method_item["api_seq"]
    mask_count, mask_locs = find_all_locations(masked_code, mask_token)
    sample_item_list = []
    for i, (loc, api) in enumerate(zip(mask_locs, api_seq)):
        if api["type"] != target_api_type:
            continue
        code_str = masked_code[:loc - 1]
        for j in range(i):
            code_str = code_str.replace(mask_token, api_seq[j]["apiName"], 1)

        if mlm:
            code_split = code_str.split()
            code_split += [mlm_token] * mlm_mask_len
            tmp_str = " ".join(code_split)
            tmp_tokens = tokenizer.encode(tmp_str)
            if len(tmp_tokens) > 1024:
                continue

            while True:
                tmp_str = " ".join(code_split)
                tmp_tokens = tokenizer.encode(tmp_str)
                if len(tmp_tokens) > max_token_len:
                    code_split = code_split[10:]
                else:
                    break
            code_str = " ".join(code_split)

        tokens = tokenizer(code_str)["input_ids"]
        if len(tokens) > max_token_len:
            continue
        api_name = api["className"] + "." + api["apiName"]
        if api_name not in api_vocab:
            api_name = api_vocab.unk_token
            api_id = api_vocab[api_name]
            candidate_ids = [api_id]
        else:
            api_id = api_vocab[api_name]
            candidate_ids = api_vocab.get_class_api(api["className"])
        target = candidate_ids.index(api_id)
        sample_item = {
            "method_id": str(method_item["method_id"]),
            "api_no": i,
            "code_str": code_str,
            "candidates": candidate_ids,
            "target": target
        }
        sample_item_list.append(sample_item)
    return sample_item_list


def prepare_all_sample(id_path: str,
                       save_path: str,
                       tokenizer: PreTrainedTokenizer,
                       api_vocab: ApiVocabulary,
                       target_api_type: str = "android",
                       max_token_len: int = 1024,
                       mlm: bool = False,
                       mlm_token: str = "<mask>",
                       mlm_mask_len: int = 5):
    with open(id_path, "r") as fp:
        method_ids = json.load(fp)
    all_sample_list = []
    for mid in tqdm(method_ids):
        sample_list = prepare_single_sample(mid,
                                            tokenizer,
                                            api_vocab,
                                            target_api_type=target_api_type,
                                            max_token_len=max_token_len,
                                            mlm=mlm,
                                            mlm_token=mlm_token,
                                            mlm_mask_len=mlm_mask_len)
        all_sample_list += sample_list
    with open(save_path, "w+") as fp:
        for line in all_sample_list:
            fp.write(json.dumps(line) + "\n")
    print("OK! Success = {}.".format(len(all_sample_list)))


def prepare():
    METHOD_ID_PATH = f""
    DATASET_PATH = f""
    PRE_TRAINED_PATH = f""
    API_VOCAB_PATH = f""
    API_TYPE = f""
    MAX_TOKEN_LEN = 512
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_PATH)
    api_vocab = ApiVocabulary.from_json(API_VOCAB_PATH)
    prepare_all_sample(id_path=METHOD_ID_PATH,
                       save_path=DATASET_PATH,
                       tokenizer=tokenizer,
                       api_vocab=api_vocab,
                       target_api_type=API_TYPE,
                       max_token_len=MAX_TOKEN_LEN)


if __name__ == "__main__":
    prepare()
