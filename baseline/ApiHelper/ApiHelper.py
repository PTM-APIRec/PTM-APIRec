import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class ApiHelper(nn.Module):
    def __init__(self,
                 api_vocab_size: int,
                 api_embed_dim: int,
                 class_vocab_size: int,
                 class_embed_dim: int,
                 hidden_size: int,
                 dropout: float = 0.0,
                 hole_id: int = 2):
        super(ApiHelper, self).__init__()
        self._api_embedding = ApiEmbeddingLayer(api_vocab_size,
                                                api_embed_dim,
                                                class_vocab_size,
                                                class_embed_dim)
        self._rnn = nn.LSTM(input_size=api_embed_dim + class_embed_dim,
                            hidden_size=hidden_size,
                            batch_first=True)
        self._linear = nn.Linear(hidden_size, api_vocab_size)
        self._dropout = nn.Dropout(dropout)
        self._hole_id = hole_id
        self._api_vocab_size = api_vocab_size

    def forward(self, inputs):
        api_ids, class_ids, candidate_ids = inputs
        hole_loc = (api_ids == self._hole_id).nonzero()[:, 1]
        api_embed = self._api_embedding(class_ids, api_ids)
        output, _ = self._rnn(api_embed)
        output = output[range(len(hole_loc)), hole_loc]
        output = self._dropout(output)
        output = self._linear(output)
        mask = Variable(candidate_ids == 0, requires_grad=False)
        output = output.masked_fill(mask, -1e9)
        return output


class ApiEmbeddingLayer(nn.Module):
    def __init__(self,
                 api_vocab_size: int,
                 api_embed_dim: int,
                 class_vocab_size: int,
                 class_embed_dim: int):
        super(ApiEmbeddingLayer, self).__init__()
        self._api_embedding = nn.Embedding(api_vocab_size, api_embed_dim, padding_idx=0)
        self._class_embedding = nn.Embedding(class_vocab_size, class_embed_dim, padding_idx=0)

        self._api_embed_dim = api_embed_dim
        self._class_embed_dim = class_embed_dim
        self._final_embed_dim = api_embed_dim + class_embed_dim

    def forward(self, class_ids, api_ids):
        class_embed = self._class_embedding(class_ids)
        api_embed = self._api_embedding(api_ids)
        final_embed = torch.cat((class_embed, api_embed), dim=-1)
        return final_embed * math.sqrt(self._final_embed_dim)
