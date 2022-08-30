import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ignite.utils import convert_tensor

from argparse import Namespace
from datetime import datetime
import os
from operator import add
from collections import Counter


class Test:
    def __init__(self,
                 dataset: dict,
                 model: nn.Module,
                 args: Namespace = None,
                 topk: int = 10,
                 **kwargs):
        self.dataset = dataset
        self.model = model
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.topk = topk
        self.model_name = kwargs.get("model_name", str(datetime.now()))

    def test(self, stat_dir: str = None):
        test_log_dir = self.args.test_dir
        test_log_path = os.path.join(test_log_dir, self.args.model_sig, self.model_name)
        writer = SummaryWriter(test_log_path)
        test_loaders = {
            ds_name: DataLoader(ds, batch_size=self.args.batch_size)
            for ds_name, ds in self.dataset.items()
        }

        all_top_k = [0 for _ in range(self.topk)]
        total_mrr = 0.0
        total = 0

        all_api_t1_acc = Counter()
        all_api_t1_err = Counter()

        self.model.to(self.device)
        self.model.eval()
        for ds_name, test_loader in test_loaders.items():
            ds_len = len(self.dataset[ds_name])
            total += ds_len
            ds_top_k = [0 for _ in range(self.topk)]
            ds_mrr = 0.0
            ds_api_t1_acc = Counter()
            ds_api_t1_err = Counter()
            with torch.no_grad():
                for data in tqdm(test_loader):
                    if self.args.model_type == "t5":
                        inputs = (data["input_ids"], data["attention_mask"], data["candidates"], data["labels"])
                    else:
                        inputs = (data["input_ids"], data["attention_mask"], data["candidates"])
                    target = data["target"]
                    batch_size = target.shape[0]
                    ans = data["ans"]
                    inputs = convert_tensor(inputs, device=self.device)
                    targets = convert_tensor(target, device=self.device)
                    outputs = self.model(inputs)

                    targets = targets.unsqueeze(1)

                    _, idx_loc = torch.sort(outputs, dim=1, descending=True)
                    hit_loc = (idx_loc == targets).nonzero()[:, 1]

                    for i in range(batch_size):
                        if hit_loc[i] == 0:
                            all_api_t1_acc[int(ans[i])] += 1
                            ds_api_t1_acc[int(ans[i])] += 1
                        else:
                            all_api_t1_err[int(ans[i])] += 1
                            ds_api_t1_err[int(ans[i])] += 1

                    print(hit_loc.tolist())
                    for k in range(self.topk):
                        ds_top_k[k] += int(torch.sum(hit_loc <= k))

                    hit_list = hit_loc.tolist()
                    hit_list = [1 / (x + 1) for x in hit_list]
                    total_mrr += sum(hit_list)
                    ds_mrr += sum(hit_list)

            print(f"Dataset: {ds_name}, len: {ds_len}, top-k: {[correct / ds_len for correct in ds_top_k]}, "
                  f"mrr: {ds_mrr / ds_len}")
            for k in range(self.topk):
                ds_top_k_value = ds_top_k[k] / ds_len
                writer.add_scalar(f"Top-k-{ds_name}", ds_top_k_value, k + 1)
            if stat_dir is not None:
                with open(os.path.join(stat_dir, ds_name + "_acc.json"), "w+") as fp:
                    json.dump(ds_api_t1_acc, fp=fp)
                with open(os.path.join(stat_dir, ds_name + "_err.json"), "w+") as fp:
                    json.dump(ds_api_t1_err, fp=fp)
            all_top_k = list(map(add, all_top_k, ds_top_k))

        print(f"Total: {total}, top-k: {[correct / total for correct in all_top_k]}, mrr: {total_mrr / total}")
        for k in range(self.topk):
            topk_value = all_top_k[k] / total
            writer.add_scalar("Top-k-All", topk_value, k + 1)
        writer.add_scalar("MRR", total_mrr / total)
        if stat_dir is not None:
            with open(os.path.join(stat_dir, "all_acc.json"), "w+") as fp:
                json.dump(all_api_t1_acc, fp=fp)
            with open(os.path.join(stat_dir, "all_err.json"), "w+") as fp:
                json.dump(all_api_t1_err, fp=fp)

        writer.close()
