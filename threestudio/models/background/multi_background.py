from dataclasses import dataclass, field
from typing import Optional, Tuple

import threestudio
import torch.nn as nn
from threestudio.models.background.base import BaseBackground


@threestudio.register("neural-environment-map-background-multi")
class NeuralEnvironmentMapBackgroundMulti(BaseBackground):
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
        n_particles: int = 6

    cfg: Config

    def configure(self) -> None:
        self.n_particles = self.cfg.n_particles
        multi_part_cfg = self.cfg.copy()
        del multi_part_cfg.n_particles
        self.multi_background = []
        for _ in range(self.n_particles):
            self.multi_background.append(threestudio.find("neural-environment-map-background")(cfg=multi_part_cfg))
        self.multi_background = nn.ModuleList(self.multi_background)

    def forward(self, index: int, dirs):
        return self.multi_background[index](dirs)
