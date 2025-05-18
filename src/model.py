from dataclasses import dataclass

from torch import nn


@dataclass
class BertConfig:
    vocab_size: int = 30522
    d_model: int = 768
    n_hidden_size: int = 3072
    n_types: int = 2
    block_size: int = 512
    num_layers: int = 12
    n_head: int = 12


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self = nn.ModuleDict(
            {
                "query": nn.Linear(config.d_model, config.d_model, bias=True),
                "key": nn.Linear(config.d_model, config.d_model, bias=True),
                "value": nn.Linear(config.d_model, config.d_model, bias=True),
            }
        )
        self.output = nn.ModuleDict(
            {
                "dense": nn.Linear(config.d_model, config.d_model, bias=True),
                "LayerNorm": nn.LayerNorm(config.d_model),
            }
        )

    def forward(self, x, attention_mask=None):
        batch, block_size, d_model = x.size()
        n_head = self.config.n_head
        d_model_per_head = d_model // n_head
        q = self.self["query"](x)
        k = self.self["key"](x)
        v = self.self["value"](x)
        q = q.view(batch, block_size, n_head, d_model_per_head).transpose(
            1, 2
        )  # [batch, n_head, block_size, d_model_per_head]
        k = k.view(batch, block_size, n_head, d_model_per_head).transpose(1, 2)
        v = v.view(batch, block_size, n_head, d_model_per_head).transpose(1, 2)
        scores = q @ k.transpose(2, 3) / (d_model_per_head**0.5)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        rescaled_scores = scores - scores.max(dim=-1).values.unsqueeze(-1)
        attention = nn.functional.softmax(rescaled_scores, dim=-1)
        H = attention @ v
        H = H.transpose(1, 2).contiguous().view(batch, block_size, d_model)
        H = self.output["dense"](H)
        H = self.output["LayerNorm"](x + H)
        return H


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Self-attention components
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.ModuleDict(
            {
                "dense": nn.Linear(config.d_model, config.n_hidden_size, bias=True),
            }
        )
        self.output = nn.ModuleDict(
            {
                "dense": nn.Linear(config.n_hidden_size, config.d_model, bias=True),
                "LayerNorm": nn.LayerNorm(config.d_model),
            }
        )

    def forward(self, hidden_states, attention_mask):
        pass


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(
            {
                "word_embeddings": nn.Embedding(config.vocab_size, config.d_model),
                "position_embeddings": nn.Embedding(config.block_size, config.d_model),
                "token_type_embeddings": nn.Embedding(config.n_types, config.d_model),
                "LayerNorm": nn.LayerNorm(config.d_model),
            }
        )
        self.encoder = nn.ModuleDict(
            {
                "layer": nn.ModuleList(
                    [EncoderLayer(config) for _ in range(config.num_layers)]
                )
            }
        )
        self.pooler = nn.ModuleDict(
            {
                "dense": nn.Linear(config.d_model, config.d_model, bias=True),
            }
        )

    def forward(self, input_ids, attention_mask):
        pass
