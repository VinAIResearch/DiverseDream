from dataclasses import dataclass

from jaxtyping import Float
from threestudio.utils.base import BaseModule
from torch import Tensor


class BaseBackground(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pass

    cfg: Config

    def configure(self):
        pass

    def forward(self, dirs: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        raise NotImplementedError
