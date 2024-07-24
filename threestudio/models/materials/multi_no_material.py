from dataclasses import dataclass
from typing import Any, Dict, Optional

import threestudio
import torch.nn as nn

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float
from threestudio.models.materials.base import BaseMaterial

# PyTorch Tensor type
from torch import Tensor


@threestudio.register("no-material-multi")
class NoMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        input_feature_dims: Optional[int] = None
        mlp_network_config: Optional[dict] = None
        requires_normal: bool = False
        n_particles: int = 6

    cfg: Config

    def configure(self) -> None:
        self.n_particles = self.cfg.n_particles
        multi_part_cfg = self.cfg.copy()
        del multi_part_cfg.n_particles
        self.multi_material = []
        for _ in range(self.n_particles):
            self.multi_material.append(threestudio.find("no-material")(cfg=multi_part_cfg))
        self.multi_material = nn.ModuleList(self.multi_material)

    def forward(self, index: int, features: Float[Tensor, "..."], **kwargs) -> Float[Tensor, "..."]:
        return self.multi_material[index](features, **kwargs)

    def export(self, index: int, features: Float[Tensor, "..."], **kwargs) -> Dict[str, Any]:
        return self.multi_material[index].export(features, **kwargs)
