import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "alpaca_sft.prompt"


class PackedSFTDataset(Dataset):
    def __init__(self, tokens: list[int], n: int, length: int) -> None:
        self._tokens = tokens
        self._n = n
        self._length = length

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self._length
        input_ids = torch.tensor(self._tokens[start : start + self._length], dtype=torch.long)
        labels = torch.tensor(self._tokens[start + 1 : start + self._length + 1], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    with open(_TEMPLATE_PATH) as f:
        template = f.read().rstrip("\n")

    examples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if shuffle:
        random.shuffle(examples)

    all_tokens: list[int] = []
    for ex in examples:
        text = template.format(instruction=ex["prompt"], response=ex["response"])
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.append(tokenizer.bos_token_id)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)

    n_examples = (len(all_tokens) - 1) // seq_length
    return PackedSFTDataset(all_tokens, n_examples, seq_length)