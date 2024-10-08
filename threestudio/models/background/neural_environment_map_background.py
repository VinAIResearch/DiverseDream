import random
from dataclasses import dataclass, field

import threestudio
import torch
from threestudio.models.background.base import BaseBackground
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation


"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects
2. jaxtyping (https://github.com/google/jaxtyping/blob/main/API.md) for PyTorch tensors

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
2. Runtime type checking with typeguard (install with pip and triggered at runtime, mainly for tensor dtype and shape checking)
"""

# Basic types
from typing import Optional, Tuple

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float

# PyTorch Tensor type
from torch import Tensor


@threestudio.register("neural-environment-map-background")
class NeuralEnvironmentMapBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        dir_encoding_config: dict = field(default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3})
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        )
        random_aug: bool = False
        random_aug_prob: float = 0.5
        eval_color: Optional[Tuple[float, float, float]] = None

    cfg: Config

    def configure(self) -> None:
        self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
        self.network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.n_output_dims,
            self.cfg.mlp_network_config,
        )

    def forward(self, dirs: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        if not self.training and self.cfg.eval_color is not None:
            return torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(dirs) * torch.as_tensor(
                self.cfg.eval_color
            ).to(dirs)
        # viewdirs must be normalized before passing to this function
        dirs = (dirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, 3))
        color = self.network(dirs_embd).view(*dirs.shape[:-1], self.cfg.n_output_dims)
        color = get_activation(self.cfg.color_activation)(color)
        if self.training and self.cfg.random_aug and random.random() < self.cfg.random_aug_prob:
            # use random background color with probability random_aug_prob
            color = color * 0 + (  # prevent checking for unused parameters in DDP
                torch.rand(dirs.shape[0], 1, 1, self.cfg.n_output_dims).to(dirs).expand(*dirs.shape[:-1], -1)
            )
        return color
