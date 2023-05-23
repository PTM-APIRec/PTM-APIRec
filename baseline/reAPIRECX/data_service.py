import random
import numpy as np

# split dataset: train and validation
# same with ours, select 94% as train and the rest as validation
from collections import defaultdict

from tqdm import tqdm

all_projects = []

def split_dataset(train_rate):
    # with open("all_projects.txt", "r")
    train_num = int(train_rate * len(all_projects))
    train_projects = random.choice(all_projects, train_num)
    return train_projects


def generate_search_class_api_dict():
    file_name = '../output/api_ori_train_without_project.txt'
    class_api_dict = defaultdict(list)
    with open(file_name, "r") as reader:
        for seq in tqdm(reader.readlines()):
            apis = seq.split(" ")
            for api in apis:
                if api.find(".") == -1:
                    continue
                # assert len(api.split(".")) == 2
                if len(api.split(".")) != 2:
                    # print(api)
                    # print(api.split("."))
                    continue
                class_name = api.split(".")[0]
                api_name = api.split(".")[1].strip("\n")
                class_api_dict[class_name].append(api_name)
    print(len(class_api_dict.keys()))
    np.save('search_class_api_dict.npy', class_api_dict)


if __name__ == "__main__":
    generate_search_class_api_dict()




