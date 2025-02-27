from abc import ABC, abstractmethod
from typing import Dict, NamedTuple
import torch
import torch.nn as nn
from gaussian_splatting.utils import build_rotation
from gaussian_splatting.gaussian_model import GaussianModel

from .trainer import AbstractTrainer, BaseTrainer, TrainerWrapper
from .opacity_reset import OpacityResetTrainer


class DensificationParams(NamedTuple):
    new_xyz: torch.Tensor = None
    new_features_dc: torch.Tensor = None
    new_features_rest: torch.Tensor = None
    new_opacities: torch.Tensor = None
    new_scaling: torch.Tensor = None
    new_rotation: torch.Tensor = None
    remove_mask: torch.Tensor = None


class AbstractDensifier(ABC):

    @abstractmethod
    def update_densification_stats(self, out):
        raise NotImplementedError

    @abstractmethod
    def densify_and_prune(self) -> DensificationParams:
        raise NotImplementedError


class Densifier(AbstractDensifier):

    def __init__(self, model: GaussianModel, scene_extent,
                 percent_dense=0.01,
                 densify_grad_threshold=0.0002,
                 densify_opacity_threshold=0.005,
                 prune_from_iter=1000,
                 prune_screensize_threshold=20,
                 device=None):
        self.model = model
        self.scene_extent = scene_extent
        self.percent_dense = percent_dense
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_opacity_threshold = densify_opacity_threshold
        self.prune_screensize_threshold = prune_screensize_threshold
        self.prune_from_iter = prune_from_iter
        self.update_counter = 0

        self.device = device if device is not None else model._xyz.device
        self.xyz_gradient_accum = torch.zeros((model.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((model.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((model.get_xyz.shape[0]), device=self.device)

    def update_densification_stats(self, out):
        viewspace_points, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
        self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_points.grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1
        self.update_counter += 1

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.model.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.model.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.model.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.model.scaling_inverse_activation(self.model.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self.model._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self.model._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self.model._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self.model._opacity[selected_pts_mask].repeat(N, 1)

        return DensificationParams(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacities=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
            remove_mask=selected_pts_mask
        )

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self.model._xyz[selected_pts_mask]
        new_features_dc = self.model._features_dc[selected_pts_mask]
        new_features_rest = self.model._features_rest[selected_pts_mask]
        new_opacities = self.model._opacity[selected_pts_mask]
        new_scaling = self.model._scaling[selected_pts_mask]
        new_rotation = self.model._rotation[selected_pts_mask]

        return DensificationParams(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacities=new_opacities,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
        )

    def densify_and_prune(self):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        clone = self.densify_and_clone(grads, self.densify_grad_threshold, self.scene_extent)
        split = self.densify_and_split(grads, self.densify_grad_threshold, self.scene_extent)

        remove_mask = split.remove_mask
        if self.update_counter > self.prune_from_iter:
            prune_mask = (self.model.get_opacity < self.densify_opacity_threshold).squeeze()
            big_points_vs = self.max_radii2D > self.prune_screensize_threshold
            big_points_ws = self.model.get_scaling.max(dim=1).values > 0.1 * self.scene_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

            remove_mask = torch.logical_or(prune_mask, split.remove_mask)

        new_size = self.model.get_xyz.shape[0] + clone.new_xyz.shape[0] + split.new_xyz.shape[0] - remove_mask.sum()
        self.xyz_gradient_accum = torch.zeros((new_size, 1), device=self.device)
        self.denom = torch.zeros((new_size, 1), device=self.device)
        self.max_radii2D = torch.zeros((new_size), device=self.device)

        return DensificationParams(
            new_xyz=torch.cat((clone.new_xyz, split.new_xyz), dim=0),
            new_features_dc=torch.cat((clone.new_features_dc, split.new_features_dc), dim=0),
            new_features_rest=torch.cat((clone.new_features_rest, split.new_features_rest), dim=0),
            new_opacities=torch.cat((clone.new_opacities, split.new_opacities), dim=0),
            new_scaling=torch.cat((clone.new_scaling, split.new_scaling), dim=0),
            new_rotation=torch.cat((clone.new_rotation, split.new_rotation), dim=0),
            remove_mask=remove_mask
        )


def cat_tensors_to_optimizer(optimizer: torch.optim.Optimizer, tensors_dict: Dict[str, torch.Tensor]):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        extension_tensor = tensors_dict[group["name"]]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:

            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def mask_tensors_in_optimizer(optimizer: torch.optim.Optimizer, prune_mask: torch.Tensor):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        mask = ~prune_mask
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:

            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


class DensificationTrainer(BaseTrainer):
    '''
    Trainer that densifies the model.
    !This class must inherent from BaseTrainer rather than TrainerWrapper, since it should modify the tensors in the optimizer.
    '''

    def __init__(
            self, model: GaussianModel,
            spatial_lr_scale: float,
            densifier: AbstractDensifier,
            densify_from_iter: int = 500,
            densify_until_iter: int = 15000,
            densification_interval: int = 100,
            *args, **kwargs
    ):
        super().__init__(model, spatial_lr_scale, *args, **kwargs)
        self.densifier = densifier
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval

    def add_points(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        optimizable_tensors = cat_tensors_to_optimizer(self.optimizer, {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation})

        self.model.update_points_add(
            xyz=optimizable_tensors["xyz"],
            features_dc=optimizable_tensors["f_dc"],
            features_rest=optimizable_tensors["f_rest"],
            opacity=optimizable_tensors["opacity"],
            scaling=optimizable_tensors["scaling"],
            rotation=optimizable_tensors["rotation"],
        )

    def remove_points(self, rm_mask):
        optimizable_tensors = mask_tensors_in_optimizer(self.optimizer, rm_mask)

        self.model.update_points_remove(
            removed_mask=rm_mask,
            xyz=optimizable_tensors["xyz"],
            features_dc=optimizable_tensors["f_dc"],
            features_rest=optimizable_tensors["f_rest"],
            opacity=optimizable_tensors["opacity"],
            scaling=optimizable_tensors["scaling"],
            rotation=optimizable_tensors["rotation"],
        )

    def densify_and_prune(self):
        params = self.densifier.densify_and_prune()
        if params.remove_mask is not None:
            self.remove_points(params.remove_mask)
        if params.new_xyz is not None:
            assert params.new_features_dc is not None
            assert params.new_features_rest is not None
            assert params.new_opacities is not None
            assert params.new_scaling is not None
            assert params.new_rotation is not None
            self.add_points(
                params.new_xyz,
                params.new_features_dc,
                params.new_features_rest,
                params.new_opacities,
                params.new_scaling,
                params.new_rotation)
        torch.cuda.empty_cache()

    def before_optim_hook(self, loss, out, camera):
        with torch.no_grad():
            if self.curr_step < self.densify_until_iter:
                self.densifier.update_densification_stats(out)
            if self.densify_from_iter <= self.curr_step < self.densify_until_iter and self.curr_step % self.densification_interval == 0:
                self.densify_and_prune()


def BaseDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        densify_from_iter=500,
        densify_until_iter=15000,
        densification_interval=100,
        opacity_reset_interval=3000,
        percent_dense=0.01,
        densify_grad_threshold=0.0002,
        densify_opacity_threshold=0.005,
        prune_from_iter=1000,
        prune_screensize_threshold=20,
        device=None,
        *args, **kwargs):
    return OpacityResetTrainer(
        DensificationTrainer(
            model, scene_extent,
            Densifier(model, scene_extent, percent_dense,
                      densify_grad_threshold, densify_opacity_threshold,
                      prune_from_iter, prune_screensize_threshold,
                      device=device),
            densify_from_iter, densify_until_iter, densification_interval,
            *args, **kwargs
        ),
        opacity_reset_until_iter=densify_until_iter,
        opacity_reset_interval=opacity_reset_interval
    )
