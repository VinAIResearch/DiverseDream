from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import threestudio
import torch

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float

# from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from threestudio.utils.ops import shifted_expotional_decay

# PyTorch Tensor type
from torch import Tensor


@dataclass
class PromptProcessorOutputCustom:
    text_embeddings: Float[Tensor, "..."]
    uncond_text_embeddings: Float[Tensor, "..."]
    text_embeddings_vd: Float[Tensor, "..."]
    uncond_text_embeddings_vd: Float[Tensor, "..."]
    directions: List[Any]
    direction2idx: Dict[str, int]
    use_perp_neg: bool
    perp_neg_f_sb: Tuple[float, float, float]
    perp_neg_f_fsb: Tuple[float, float, float]
    perp_neg_f_fs: Tuple[float, float, float]
    perp_neg_f_sf: Tuple[float, float, float]
    prompt: str
    prompts_vd: List[str]
    hiper_scale: float = 0.9

    def get_text_embeddings(
        self,
        elevation: Float[Tensor, "..."],
        azimuth: Float[Tensor, "..."],
        camera_distances: Float[Tensor, "..."],
        num_train_step,
        view_dependent_prompting: bool = True,
        hiper_guidance=None,
        learnable_text=None,
    ) -> Float[Tensor, "..."]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[d.condition(elevation, azimuth, camera_distances)] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(batch_size, -1, -1)  # type: ignore
        scale = self.hiper_scale

        if hiper_guidance is not None:
            n_hiper = hiper_guidance.shape[1]
            text_embeddings_hiper = torch.concat(
                [text_embeddings[:, :-n_hiper], hiper_guidance.expand(text_embeddings.shape[0], -1, -1) * scale], dim=1
            )
            text_embeddings = text_embeddings_hiper
            if learnable_text is not None:
                n_learnable = learnable_text.shape[1]
                text_embeddings = torch.concat(
                    [
                        text_embeddings[:, : -n_hiper - n_learnable],
                        learnable_text.expand(text_embeddings.shape[0], -1, -1),
                        text_embeddings[:, -n_hiper:],
                    ],
                    dim=1,
                )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)

    def get_text_embeddings_perp_neg(
        self,
        elevation: Float[Tensor, "..."],
        azimuth: Float[Tensor, "..."],
        camera_distances: Float[Tensor, "..."],
        view_dependent_prompting: bool = True,
        hiper_guidance=None,
        learnable_text=None,
    ) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
        assert view_dependent_prompting, "Perp-Neg only works with view-dependent prompting"

        batch_size = elevation.shape[0]

        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in self.directions:
            direction_idx[d.condition(elevation, azimuth, camera_distances)] = self.direction2idx[d.name]
        # 0 - side view
        # 1 - front view
        # 2 - back view
        # 3 - overhead view

        pos_text_embeddings = []
        neg_text_embeddings = []
        neg_guidance_weights = []
        uncond_text_embeddings = []

        side_emb = self.text_embeddings_vd[0]
        front_emb = self.text_embeddings_vd[1]
        back_emb = self.text_embeddings_vd[2]
        overhead_emb = self.text_embeddings_vd[3]

        for idx, _ele, azi, _dis in zip(direction_idx, elevation, azimuth, camera_distances):
            azi = shift_azimuth_deg(azi)  # to (-180, 180)
            uncond_text_embeddings.append(self.uncond_text_embeddings_vd[idx])  # should be ""
            if idx.item() == 3:  # overhead view
                pos_text_embeddings.append(overhead_emb)  # side view
                # dummy
                neg_text_embeddings += [
                    self.uncond_text_embeddings_vd[idx],
                    self.uncond_text_embeddings_vd[idx],
                ]
                neg_guidance_weights += [0.0, 0.0]
            else:  # interpolating views
                if torch.abs(azi) < 90:
                    # front-side interpolation
                    # 0 - complete side, 1 - complete front
                    r_inter = 1 - torch.abs(azi) / 90
                    pos_text_embeddings.append(r_inter * front_emb + (1 - r_inter) * side_emb)
                    neg_text_embeddings += [front_emb, side_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_fs, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_sf, 1 - r_inter),
                    ]
                else:
                    # side-back interpolation
                    # 0 - complete back, 1 - complete side
                    r_inter = 2.0 - torch.abs(azi) / 90
                    pos_text_embeddings.append(r_inter * side_emb + (1 - r_inter) * back_emb)
                    neg_text_embeddings += [side_emb, front_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_sb, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_fsb, r_inter),
                    ]

        text_embeddings = torch.cat(
            [
                torch.stack(pos_text_embeddings, dim=0),
                torch.stack(uncond_text_embeddings, dim=0),
                torch.stack(neg_text_embeddings, dim=0),
            ],
            dim=0,
        )

        return text_embeddings, torch.as_tensor(neg_guidance_weights, device=elevation.device).reshape(batch_size, 2)


def shift_azimuth_deg(azimuth: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    # shift azimuth angle (in degrees), to [-180, 180]
    return (azimuth + 180) % 360 - 180


@threestudio.register("stable-diffusion-prompt-processor-tsd")
class StableDiffusionPromptProcessorCustom(StableDiffusionPromptProcessor):
    @dataclass
    class Config(StableDiffusionPromptProcessor.Config):
        hiper_scale: float = 0.9

    cfg: Config

    def __call__(self) -> PromptProcessorOutputCustom:
        return PromptProcessorOutputCustom(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            prompt=self.prompt,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            prompts_vd=self.prompts_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
            hiper_scale=self.cfg.hiper_scale,
        )
