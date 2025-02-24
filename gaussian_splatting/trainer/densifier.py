from typing import Dict
import torch
import torch.nn as nn
from gaussian_splatting.utils import build_rotation
from gaussian_splatting.gaussian_model import GaussianModel

from .trainer import AbstractTrainer, BaseTrainer, TrainerWrapper


class _Densifier:
    def __init__(self,
                 model: GaussianModel, optimizer: torch.optim.Optimizer,
                 percent_dense=0.01,
                 device=None):
        self.model = model
        self.optimizer = optimizer
        self.percent_dense = percent_dense
        device = device if device is not None else model._xyz.device
        self.xyz_gradient_accum = torch.zeros((model.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((model.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((model.get_xyz.shape[0]), device=device)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def update_densification_stats(self, radii, viewspace_points, visibility_filter):
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
        self.add_densification_stats(viewspace_points, visibility_filter)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.model.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.model.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

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

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

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

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.model.get_xyz.shape[0]), device="cuda")

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def reset_opacity(self):
        opacities_new = self.model.inverse_opacity_activation(torch.min(self.model.get_opacity, torch.ones_like(self.model.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.model._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


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


class Densifier(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            densify_from_iter: int,
            densify_until_iter: int,
            densification_interval: int,
            opacity_reset_interval: int,
            scene_extent: float,
            densify_grad_threshold=0.0002,
            percent_dense=0.01
    ):
        super().__init__(base_trainer)
        self.densifier = _Densifier(self.model, self.optimizer)
        self.scene_extent = scene_extent
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.percent_dense = percent_dense

    def add_points(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        optimizable_tensors = cat_tensors_to_optimizer(self.optimizer, {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation})

        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]

    def remove_points(self, rm_mask):
        optimizable_tensors = mask_tensors_in_optimizer(self.optimizer, rm_mask)

        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]

    def step(self, camera):
        self.update_learning_rate()
        loss, out = self.forward_backward(camera)
        viewspace_points, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
        with torch.no_grad():
            if self.curr_step < self.densify_until_iter:
                self.densifier.update_densification_stats(radii, viewspace_points, visibility_filter)
                if self.curr_step >= self.densify_from_iter and self.curr_step % self.densification_interval == 0:
                    size_threshold = 20 if self.curr_step > self.opacity_reset_interval else None
                    self.densifier.densify_and_prune(self.densify_grad_threshold, 0.005, self.scene_extent, size_threshold)
                if self.curr_step % self.opacity_reset_interval == 0:
                    self.densifier.reset_opacity()
        self.optim_step()
        return loss, out


def DensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        densify_from_iter=500,
        densify_until_iter=15000,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_grad_threshold=0.0002,
        percent_dense=0.01,
        *args, **kwargs):
    return Densifier(
        BaseTrainer(model, scene_extent, *args, **kwargs),
        densify_from_iter, densify_until_iter, densification_interval, opacity_reset_interval, scene_extent, densify_grad_threshold, percent_dense
    )
