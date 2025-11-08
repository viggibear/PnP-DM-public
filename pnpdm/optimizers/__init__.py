import torch
from typing import Iterator
from torch.nn.parameter import Parameter

def get_optimizer(params: Iterator[Parameter], name: str, **kwargs):
    if name == 'adamw':
        return torch.optim.AdamW(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")