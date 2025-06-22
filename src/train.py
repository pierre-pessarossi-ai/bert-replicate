import math
import time
import numpy as np
import torch
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

torch.set_float32_matmul_precision("high")

bert_config = BertConfig(num_layers=1)
model = BertForMaskedLM(bert_config).to(device)

if torch.cuda.is_available():
    model = torch.compile(model)

# Count model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


BATCH_SIZE = 512
STEP_BATCH_SIZE = 128
number_grad_accum_steps = BATCH_SIZE // STEP_BATCH_SIZE

assert BATCH_SIZE % STEP_BATCH_SIZE == 0

dataset = ShakespeareDataset(
    "data/tokenized_train.npy", 64, STEP_BATCH_SIZE, vocab_size=bert_config.vocab_size
)


def get_lr(
    step: int, warmup_steps: int, max_steps: int, min_lr: float, max_lr: float
) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps * max_lr
    elif step > max_steps:
        return min_lr
    else:
        step = step - warmup_steps
        step_decay = max_steps - warmup_steps
        decay_ratio = step / step_decay
        assert 0 <= decay_ratio <= 1
        coef = 1 / 2 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * coef


def main():
    max_steps = 100
    warm_up_steps = 10
    min_lr = 3e-6
    max_lr = 1e-5
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8
    )
    for step in range(max_steps + 1):
        t0 = time.time()
        loss_accum = 0
        for _ in range(number_grad_accum_steps):
            x, y = dataset.next_batch()
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, bert_config.vocab_size), y.view(-1), ignore_index=-100
            )
            loss /= number_grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, warm_up_steps, max_steps, min_lr, max_lr)
        for params in optimizer.param_groups:
            params["lr"] = lr
        optimizer.step()
        optimizer.zero_grad()
        t1 = time.time()
        tokens_per_sec = (
            dataset.batch_size
            * dataset.context_length
            * number_grad_accum_steps
            / (t1 - t0)
        )
        if step % 10 == 0 or step == max_steps:
            print(
                f"LR: {lr * 1e5:.4f}e-5, Step {step} loss: {loss_accum}, grad_norm: {norm.item():.4f}, time: {t1 - t0:.2f}s, tokens/sec: {tokens_per_sec:.2f}"
            )


if __name__ == "__main__":
    main()
# TODO: add gradient accumulation
# TODO: add weight decay + fused adamW
# TODO: add checkpointing with state dict and optimizer + check we can resume training
# TODO: add parellel training


# Total parameters: 110,104,890
# Trainable parameters: 110,104,890
# Step 0 loss: 10.4375, time: 19.53s, tokens/sec: 419.41
# Step 10 loss: 9.6875, time: 0.06s, tokens/sec: 129341.16
# Step 20 loss: 9.25, time: 0.06s, tokens/sec: 129397.66
# Step 30 loss: 9.1875, time: 0.06s, tokens/sec: 129363.56
# Step 40 loss: 5.03125, time: 0.06s, tokens/sec: 129346.03
# Step 50 loss: 2.71875, time: 0.06s, tokens/sec: 129366.00
# Step 60 loss: 2.515625, time: 0.06s, tokens/sec: 129221.01
# Step 70 loss: 2.4375, time: 0.06s, tokens/sec: 129353.82
# Step 80 loss: 2.296875, time: 0.06s, tokens/sec: 128920.89
# Step 90 loss: 2.0625, time: 0.06s, tokens/sec: 128563.45
# Step 100 loss: 1.9765625, time: 0.06s, tokens/sec: 128962.51
