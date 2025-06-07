import numpy as np
import torch
import torch.nn as nn
from model import BertConfig, BertForMaskedLM


def load_data(path: str) -> torch.Tensor:
    tokens = np.load(path)
    tokens = tokens.astype(np.int32)
    tokens_tensors = torch.tensor(tokens, dtype=torch.long)
    return tokens_tensors


class ShakespeareDataset:

    def __init__(
        self, path: str, context_length: int, batch_size: int, vocab_size: int
    ):
        self.tokens = load_data(path)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.batch_size = batch_size
        torch.manual_seed(42)

        self.state = 0

    def next_batch(self):
        B, T = self.batch_size, self.context_length
        seq = self.tokens[self.state : self.state + B * T]
        batch_seq = seq.view(B, T)

        # add mask and random tokens
        masked_positions = torch.rand((B, T), generator=torch.manual_seed(42)) < 0.15
        labels = batch_seq.clone()
        labels[~masked_positions] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_positions
        )
        batch_seq[indices_replaced] = 103
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_positions
            & ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        batch_seq[indices_random] = random_words[indices_random]

        self.state += B * T

        if self.state + B * T >= len(self.tokens):
            self.state = 0

        return batch_seq, labels


# add training loop

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


torch.manual_seed(1478)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1478)
elif torch.backends.mps.is_available():
    torch.mps.manual_seed(1478)

bert_config = BertConfig(num_layers=1)
model = BertForMaskedLM(bert_config).to(device)

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

dataset = ShakespeareDataset(
    "data/tokenized_train.npy", 32, 8, vocab_size=bert_config.vocab_size
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
max_steps = 100
for step in range(max_steps + 1):
    x, y = dataset.next_batch()
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, bert_config.vocab_size), y.view(-1), ignore_index=-100
    )
    loss.backward()
    optimizer.step()
    if step % 10 == 0 or step == max_steps:
        print(f"Step {step} loss: {loss.item()}")
# TODO: check initialization to see if we can get expected value for initial loss
