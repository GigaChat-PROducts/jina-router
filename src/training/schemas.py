import torch
from dataclasses import dataclass


@dataclass
class TokenizerOutput:
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor