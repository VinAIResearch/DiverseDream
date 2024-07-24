# Basic types
from typing import Tuple, Union

import nvdiffrast.torch as dr
import torch

# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Float, Integer
from torch import Tensor


class NVDiffRasterizerContext:
    def __init__(self, context_type: str, device: torch.device) -> None:
        self.device = device
        self.ctx = self.initialize_context(context_type, device)

    def initialize_context(
        self, context_type: str, device: torch.device
    ) -> Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]:
        if context_type == "gl":
            return dr.RasterizeGLContext(device=device)
        elif context_type == "cuda":
            return dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f"Unknown rasterizer context type: {context_type}")

    def vertex_transform(self, verts: Float[Tensor, "..."], mvp_mtx: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        verts_homo = torch.cat([verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1)
        return torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))

    def rasterize(
        self,
        pos: Float[Tensor, "..."],
        tri: Integer[Tensor, "..."],
        resolution: Union[int, Tuple[int, int]],
    ):
        # rasterize in instance mode (single topology)
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(
        self,
        pos: Float[Tensor, "..."],
        tri: Integer[Tensor, "..."],
        resolution: Union[int, Tuple[int, int]],
    ):
        # rasterize one single mesh under a single viewpoint
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def antialias(
        self,
        color: Float[Tensor, "..."],
        rast: Float[Tensor, "..."],
        pos: Float[Tensor, "..."],
        tri: Integer[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(
        self,
        attr: Float[Tensor, "..."],
        rast: Float[Tensor, "..."],
        tri: Integer[Tensor, "..."],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "..."]:
        return dr.interpolate(attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs)

    def interpolate_one(
        self,
        attr: Float[Tensor, "..."],
        rast: Float[Tensor, "..."],
        tri: Integer[Tensor, "..."],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "..."]:
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)
