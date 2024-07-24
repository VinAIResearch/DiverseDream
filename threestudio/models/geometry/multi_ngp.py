from dataclasses import dataclass, field
from typing import Optional, Union

import threestudio
import torch
import torch.nn as nn

# Config type
from omegaconf import DictConfig
from threestudio.models.geometry.base import BaseGeometry, BaseImplicitGeometry


@threestudio.register("implicit-volume-multi")
class ImplicitVolumeMulti(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob_magic3d"
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[str] = (
            "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        )
        finite_difference_normal_eps: float = 0.01

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = 25.0

        # 4D Gaussian Annealing
        anneal_density_blob_std_config: Optional[dict] = None
        n_particles: int = 6

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.n_particles = self.cfg.n_particles
        multi_part_cfg = self.cfg.copy()
        del multi_part_cfg.n_particles
        self.multi_encoder = []
        self.encoding = []
        self.density_network = []
        self.feature_network = []
        for _ in range(self.n_particles):
            self.multi_encoder.append(threestudio.find("implicit-volume")(cfg=multi_part_cfg))
            last_geo = self.multi_encoder[-1]
            self.encoding.append(last_geo.encoding)
            self.density_network.append(last_geo.density_network)
            self.feature_network.append(last_geo.feature_network)
        self.multi_encoder = nn.ModuleList(self.multi_encoder)
        self.encoding = nn.ModuleList(self.encoding)
        self.density_network = nn.ModuleList(self.density_network)
        self.feature_network = nn.ModuleList(self.feature_network)

    def get_activated_density(
        self,
        index: int,
        points,
        density,
    ):
        raw_density, density = self.multi_encoder[index].get_activated_density(points, density)
        return raw_density, density

    def forward(self, index: int, points, output_normal: bool = False):
        return self.multi_encoder[index](points, output_normal)

    def forward_density(self, index: int, points):
        return self.multi_encoder[index].forward_density(points)

    def forward_field(self, index: int, points):
        return self.multi_encoder[index].forward_field(points)

    def forward_level(self, index: int, field, threshold):
        return self.multi_encoder[index].forward_level(field, threshold)

    def export(self, index: int, points, **kwargs):
        return self.multi_encoder[index].export(points, **kwargs)

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "ImplicitVolumeMulti":
        if isinstance(other, ImplicitVolumeMulti):
            instance = ImplicitVolumeMulti(cfg, **kwargs)
            instance.multi_encoder.load_state_dict(other.encoding.state_dict())
            return instance
        else:
            raise TypeError(f"Cannot create {ImplicitVolumeMulti.__name__} from {other.__class__.__name__}")

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False) -> None:
        for i in range(self.n_particles):
            self.multi_encoder[i].update_step(epoch, global_step, on_load_weights)
