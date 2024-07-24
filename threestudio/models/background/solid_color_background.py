import random
from dataclasses import dataclass

# Basic types
from typing import Tuple

import threestudio
import torch
import torch.nn as nn

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float
from threestudio.models.background.base import BaseBackground

# PyTorch Tensor type
from torch import Tensor


@threestudio.register("solid-color-background")
class SolidColorBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color: Tuple = (1.0, 1.0, 1.0)
        learned: bool = False
        random_aug: bool = False
        random_aug_prob: float = 0.5

    cfg: Config

    def configure(self) -> None:
        self.env_color: Float[Tensor, "..."]
        if self.cfg.learned:
            self.env_color = nn.Parameter(torch.as_tensor(self.cfg.color, dtype=torch.float32))
        else:
            self.register_buffer("env_color", torch.as_tensor(self.cfg.color, dtype=torch.float32))

    def forward(self, dirs: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        color = torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs) * self.env_color.to(dirs)
        if self.training and self.cfg.random_aug and random.random() < self.cfg.random_aug_prob:
            # use random background color with probability random_aug_prob
            color = color * 0 + (  # prevent checking for unused parameters in DDP
                torch.rand(dirs.shape[0], 1, 1, self.cfg.n_output_dims).to(dirs).expand(*dirs.shape[:-1], -1)
            )
        return color
