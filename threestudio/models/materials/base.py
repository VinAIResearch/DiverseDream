from dataclasses import dataclass
from typing import Any, Dict

from jaxtyping import Float
from threestudio.utils.base import BaseModule
from torch import Tensor


class BaseMaterial(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config
    requires_normal: bool = False
    requires_tangent: bool = False

    def configure(self):
        pass

    def forward(self, *args, **kwargs) -> Float[Tensor, "..."]:
        raise NotImplementedError

    def export(self, *args, **kwargs) -> Dict[str, Any]:
        return {}
