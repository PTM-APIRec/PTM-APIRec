import torch
import torch.nn as nn

import re
from typing import List, Tuple


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


def replace_substring(string: str,
                      sub: str,
                      rep: str,
                      n: int = 0,
                      ignore_after: bool = False) -> str:
    loc = [m.start() for m in re.finditer(sub, string)][n]
    before = string[:loc]
    if ignore_after:
        return before + rep
    after = string[loc:]
    after = after.replace(sub, rep, 1)
    return before + after


def find_all_locations(string: str, sub: str) -> Tuple[int, List[int]]:
    count = string.count(sub)
    locs = []
    start = 0
    for i in range(count):
        loc = string.find(sub, start)
        locs.append(loc)
        start = loc + 1
    return count, locs


def find_substring(string: str, sub: str, n: int = 0):
    loc = [m.start() for m in re.finditer(sub, string)][n]
    return loc



