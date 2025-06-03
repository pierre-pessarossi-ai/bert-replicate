import numpy as np
import torch


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
