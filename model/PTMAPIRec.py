import torch
import torch.nn as nn


class PTMAPIRec(nn.Module):
    def __init__(self,
                 pre_trained_model: nn.Module,
                 hidden_size: int,
                 api_size: int,
                 api_dim: int,
                 dropout: float,
                 model_type: str):
        super().__init__()

        self.transformer = pre_trained_model
        self.hidden_size = hidden_size
        self.api_embedding = nn.Embedding(api_size, api_dim, padding_idx=0)
        self.fc = nn.Linear(hidden_size, api_dim)
        self.dropout = nn.Dropout(dropout)
        self.model_type = model_type

    def forward(self, inputs):
        model_inputs = self._prepare_model_inputs(inputs)

        candidates = self._prepare_candidates(inputs)

        transformer_outputs = self.transformer(**model_inputs)
        last_hidden_states = self._get_last_hidden_states(transformer_outputs)

        context_vector = self._get_context_vector(last_hidden_states, **model_inputs)

        outputs = self.fc(context_vector)

        candidates_emb = self.api_embedding(candidates)
        candidates_emb = self.dropout(candidates_emb)

        outputs = torch.mul(outputs.unsqueeze(1), candidates_emb)
        outputs = torch.sum(outputs, dim=2)
        outputs = outputs.masked_fill(candidates == 0, -1e9)

        return outputs

    def _prepare_model_inputs(self, inputs):
        model_inputs = {}
        if self.model_type in ["gpt", "bert"]:
            input_ids, attention_mask, candidates = inputs
        elif self.model_type == "t5":
            input_ids, attention_mask, candidates, labels = inputs
            model_inputs["labels"] = labels
            model_inputs["output_hidden_states"] = True
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask
        return model_inputs

    def _prepare_candidates(self, inputs):
        return inputs[2]

    def _get_last_hidden_states(self, transformer_outputs):
        if self.model_type == "t5":
            return transformer_outputs.decoder_hidden_states[-1]
        return transformer_outputs.last_hidden_state

    def _get_context_vector(self, last_hidden_states, **kwargs):
        if self.model_type == "gpt":
            last_token_pos = torch.sum(kwargs["attention_mask"], dim=1, dtype=torch.long) - 1
        elif self.model_type == "bert":
            last_token_pos = torch.sum(kwargs["attention_mask"], dim=1, dtype=torch.long) - 2
        elif self.model_type == "t5":
            return last_hidden_states[:, -1, :]
        else:
            raise ValueError(f"Cannot get the context vector of {self.model_type} model.")
        last_token_pos = last_token_pos.unsqueeze(1)
        last_token_pos = last_token_pos.unsqueeze(2).expand(last_token_pos.size(0),
                                                            last_token_pos.size(1),
                                                            last_hidden_states.size(2))
        return torch.gather(last_hidden_states, 1, last_token_pos).squeeze(1)

