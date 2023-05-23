import torch
from torch.utils.data import Dataset, DataLoader


class TestDataSet(Dataset):
    def __init__(self, tokenizer, max_seq_len, project_name=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = self.create_dataset(project_name=project_name)
        self.dataset_len = self.dataset['dataset_len']
        print("dataset length: ", self.dataset_len)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        input_id = self.dataset['input_ids'][index]
        input_id = torch.LongTensor(input_id)
        pred_index = self.dataset['pred_indexes'][index]
        target_index = self.dataset['target_indexes'][index]
        class_index = self.dataset['class_indexes'][index]
        return (input_id, class_index, pred_index), target_index

    def create_dataset(self, project_name=None):
        file_path = './data/test/api_bpe_' + project_name + ".txt"
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        pad_token = self.tokenizer.pad_token

        input_ids = []
        class_indexes = []  # 待预测的className开始的下标
        pred_indexes = []  # 待预测位置在seq中的下标
        target_indexes = []  # 待预测的api-str结束标签
        with open(file_path, "r") as reader:
            for seq in reader.readlines():
                seq = " ".join(self.tokenizer.tokenize(seq))
                seq = seq.strip("\n")
                # 过滤不包含api的seq
                if seq.find(".") == -1:
                    continue
                tokens = seq.split(" ")
                tokens = tokens[:self.max_seq_len-2]
                isAPI = False
                pred_index = -1
                pre_end_index = -1
                for i, token in enumerate(tokens):
                    if token == "▁.":
                        # 待补全
                        isAPI = True
                        pred_index = i + 1  # 因为之后seq会加上bos，所以索引加一
                    elif token.endswith("</t>"):
                        # api结尾
                        if isAPI:
                            input_tokens = self.tokenizer.tokenize(" ".join(tokens[:i+1]))
                            input_tokens = [bos_token] + input_tokens + [eos_token]
                            input_tokens += [pad_token] * (self.max_seq_len - len(input_tokens))
                            input_id = self.tokenizer.convert_tokens_to_ids(input_tokens)

                            input_ids.append(input_id)
                            pred_indexes.append(pred_index)
                            target_indexes.append(i + 1)
                            class_indexes.append(pre_end_index + 2)
                        isAPI = False
                        pre_end_index = i

        dataset = dict()
        assert len(input_ids) == len(pred_indexes) == len(class_indexes) == len(target_indexes)
        dataset_len = len(input_ids)
        dataset['input_ids'] = input_ids
        dataset['pred_indexes'] = pred_indexes
        dataset['target_indexes'] = target_indexes
        dataset['class_indexes'] = class_indexes
        dataset['dataset_len'] = dataset_len

        return dataset


def get_test_data_loader(tokenizer, args, project_name):
    dataset = TestDataSet(tokenizer=tokenizer,
                          max_seq_len=args.max_seq_len,
                          project_name=project_name)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    return data_loader
