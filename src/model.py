from dataclasses import dataclass

from torch import nn


@dataclass
class BertConfig:
    vocab_size: int = 30522
    n_embded: int = 768
    n_hidden_size: int = 3072
    n_types: int = 2
    block_size: int = 512
    num_layers: int = 12


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self = nn.ModuleDict(
            {
                "query": nn.Linear(config.n_embded, config.n_embded, bias=True),
                "key": nn.Linear(config.n_embded, config.n_embded, bias=True),
                "value": nn.Linear(config.n_embded, config.n_embded, bias=True),
            }
        )
        self.output = nn.ModuleDict(
            {
                "dense": nn.Linear(config.n_embded, config.n_embded, bias=True),
                "LayerNorm": nn.LayerNorm(config.n_embded),
            }
        )

    def forward(self, hidden_states, attention_mask):
        pass


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Self-attention components
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.ModuleDict(
            {
                "dense": nn.Linear(config.n_embded, config.n_hidden_size, bias=True),
            }
        )
        self.output = nn.ModuleDict(
            {
                "dense": nn.Linear(config.n_hidden_size, config.n_embded, bias=True),
                "LayerNorm": nn.LayerNorm(config.n_embded),
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
                "word_embeddings": nn.Embedding(config.vocab_size, config.n_embded),
                "position_embeddings": nn.Embedding(config.block_size, config.n_embded),
                "token_type_embeddings": nn.Embedding(config.n_types, config.n_embded),
                "LayerNorm": nn.LayerNorm(config.n_embded),
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
                "dense": nn.Linear(config.n_embded, config.n_embded, bias=True),
            }
        )

    def forward(self, input_ids, attention_mask):
        pass
