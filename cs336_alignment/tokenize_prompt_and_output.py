import torch
from typing import List
from transformers import PreTrainedTokenizer

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer
):
    tokenized_prompts = tokenizer(
        prompt_strs, padding=False, add_special_tokens=False
    )["input_ids"]
    
    tokenized_outputs = tokenizer(
        output_strs, padding=False, add_special_tokens=False
    )["input_ids"]

    concat_input_ids = []
    response_starts = []
    response_ends = []

    for tokenized_prompt, tokenized_output in zip(tokenized_prompts, tokenized_outputs):
        combined = tokenized_prompt + tokenized_output
        concat_input_ids.append(combined)

        # FIXED ALIGNMENT
        response_start = len(tokenized_prompt)
        response_end = response_start + len(tokenized_output) - 1

        response_starts.append(response_start)
        response_ends.append(response_end)

    max_len = max(len(x) for x in concat_input_ids)

    # pad
    for i in range(len(concat_input_ids)):
        pad_len = max_len - len(concat_input_ids[i])
        concat_input_ids[i] += [tokenizer.pad_token_id] * pad_len

    concat_input_ids = torch.tensor(concat_input_ids)

    input_ids = concat_input_ids[:, :-1]
    labels = concat_input_ids[:, 1:]

    # build mask
    positions = torch.arange(max_len - 1).unsqueeze(0)
    response_starts = torch.tensor(response_starts).unsqueeze(1)
    response_ends = torch.tensor(response_ends).unsqueeze(1)

    response_mask = (positions >= response_starts) & (positions <= response_ends)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }