from dataclasses import dataclass
from typing import Tuple

import threestudio
import torch.nn as nn
from threestudio.models.background.base import BaseBackground


@threestudio.register("solid-color-background-multi")
class SolidColorBackgroundMulti(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color: Tuple = (1.0, 1.0, 1.0)
        learned: bool = False
        random_aug: bool = False
        random_aug_prob: float = 0.5
        n_particles: int = 6

    cfg: Config

    def configure(self) -> None:
        self.n_particles = self.cfg.n_particles
        multi_part_cfg = self.cfg.copy()
        del multi_part_cfg.n_particles
        self.multi_background = []
        for _ in range(self.multi_background):
            self.multi_background.append(threestudio.find("solid-color-background")(cfg=multi_part_cfg))
        self.multi_background = nn.ModuleList(self.multi_background)

    def forward(self, index: int, dirs):
        return self.multi_background[index](dirs)
