import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer

dataset = load_dataset("tiny_shakespeare")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_train = np.array(tokenizer.encode(dataset["train"][0]["text"]))[1:-1]

print(f"len(tokenized_train): {len(tokenized_train)}")

unk_tokens = sum(tokenized_train == 100)
cls_tokens = sum(tokenized_train == 101)
sep_tokens = sum(tokenized_train == 102)
pad_tokens = sum(tokenized_train == 0)

print(f"unk_tokens: {unk_tokens}")
print(f"cls_tokens: {cls_tokens}")
print(f"sep_tokens: {sep_tokens}")
print(f"pad_tokens: {pad_tokens}")

print(f"start: {tokenized_train[:10]}")
print(f"end: {tokenized_train[10:]}")


np.save("data/tokenized_train.npy", tokenized_train)
