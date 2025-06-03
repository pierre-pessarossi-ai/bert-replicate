from dataclasses import dataclass

from torch import nn
import torch


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
        H = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )
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

    def forward(self, x, attention_mask=None):
        x_attention = self.attention(x, attention_mask)
        x_intermediate = self.intermediate["dense"](x_attention)
        x_intermediate = nn.functional.gelu(x_intermediate)
        x_output = self.output["dense"](x_intermediate)
        x_output = self.output["LayerNorm"](x_attention + x_output)
        return x_output


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        torch.manual_seed(42)  # Set fixed seed for reproducibility
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

    def forward(self, input_ids, attention_mask=None):
        torch.manual_seed(42)  # Reset seed before forward pass
        x = self.embeddings["word_embeddings"](input_ids)
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = x + self.embeddings["position_embeddings"](positions)
        x = x + self.embeddings["token_type_embeddings"](torch.zeros_like(input_ids))
        x = self.embeddings["LayerNorm"](x)
        for layer in self.encoder["layer"]:
            x = layer(x, attention_mask)
        x = self.pooler["dense"](x)
        return x


class BertForMaskedLM(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.decoder.weight = self.bert.embeddings["word_embeddings"].weight

    def forward(self, input_ids, attention_mask=None):
        x = self.bert(input_ids, attention_mask)
        x = self.transform(x)
        x = self.decoder(x)
        return x
