from dataclasses import dataclass

# Basic types
from typing import Any, Dict, Optional

import threestudio

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_mlp
from threestudio.utils.ops import get_activation

# PyTorch Tensor type
from torch import Tensor


@threestudio.register("no-material")
class NoMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        input_feature_dims: Optional[int] = None
        mlp_network_config: Optional[dict] = None
        requires_normal: bool = False

    cfg: Config

    def configure(self) -> None:
        self.use_network = False
        if self.cfg.input_feature_dims is not None and self.cfg.mlp_network_config is not None:
            self.network = get_mlp(
                self.cfg.input_feature_dims,
                self.cfg.n_output_dims,
                self.cfg.mlp_network_config,
            )
            self.use_network = True
        self.requires_normal = self.cfg.requires_normal

    def forward(self, features: Float[Tensor, "..."], **kwargs) -> Float[Tensor, "..."]:
        if not self.use_network:
            assert (
                features.shape[-1] == self.cfg.n_output_dims
            ), f"Expected {self.cfg.n_output_dims} output dims, only got {features.shape[-1]} dims input."
            color = get_activation(self.cfg.color_activation)(features)
        else:
            color = self.network(features.view(-1, features.shape[-1])).view(
                *features.shape[:-1], self.cfg.n_output_dims
            )
            color = get_activation(self.cfg.color_activation)(color)
        return color

    def export(self, features: Float[Tensor, "..."], **kwargs) -> Dict[str, Any]:
        color = self(features, **kwargs).clamp(0, 1)
        assert color.shape[-1] >= 3, "Output color must have at least 3 channels"
        if color.shape[-1] > 3:
            threestudio.warn("Output color has >3 channels, treating the first 3 as RGB")
        return {"albedo": color[..., :3]}
