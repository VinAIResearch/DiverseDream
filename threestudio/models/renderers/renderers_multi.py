from dataclasses import dataclass
from typing import Dict, Optional

import threestudio
import torch.nn as nn

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer

# PyTorch Tensor type
from torch import Tensor


@threestudio.register("nerf-volume-renderer-multi")
class NeRFVolumeRendererMulti(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        # in ["occgrid", "proposal", "importance"]
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for proposal
        proposal_network_config: Optional[dict] = None
        prop_optimizer_config: Optional[dict] = None
        prop_scheduler_config: Optional[dict] = None
        num_samples_per_ray_proposal: int = 64

        # for importance
        num_samples_per_ray_importance: int = 64

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.n_particles = self.geometry.n_particles

        self.geometry = geometry
        self.material = material
        self.background = background

        assert geometry.n_particles == material.n_particles == background.n_particles
        self.renderer_lst = []
        for i in range(self.n_particles):
            cur_geo = geometry.multi_encoder[i]
            cur_bg = background.multi_background[i]
            cur_mat = material.multi_material[i]
            self.renderer_lst.append(
                threestudio.find("nerf-volume-renderer")(
                    cfg=self.cfg, geometry=cur_geo, background=cur_bg, material=cur_mat
                )
            )
        self.renderer_lst = nn.ModuleList(self.renderer_lst)

    def forward(
        self,
        idx,
        rays_o: Float[Tensor, "..."],
        rays_d: Float[Tensor, "..."],
        light_positions: Float[Tensor, "..."],
        bg_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        out = self.renderer_lst[idx](rays_o, rays_d, light_positions, bg_color, **kwargs)
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False) -> None:
        for i in range(self.n_particles):
            self.renderer_lst[i].update_step(epoch, global_step, on_load_weights)

    def update_step_end(self, epoch: int, global_step: int) -> None:
        for i in range(self.n_particles):
            self.renderer_lst[i].update_step_end(epoch, global_step)

    def train(self, mode=True):
        for i in range(self.n_particles):
            self.renderer_lst[i].train(mode)
        return super().train(mode=mode)

    def eval(self):
        for i in range(self.n_particles):
            self.renderer_lst[i].eval()
        return super().eval()
