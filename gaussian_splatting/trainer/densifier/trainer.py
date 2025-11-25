from typing import Callable, Dict, List, Tuple
import torch
import torch.nn as nn
from gaussian_splatting import GaussianModel
from gaussian_splatting.trainer import BaseTrainer

from .abc import AbstractDensifier


def cat_tensors_to_optimizer(optimizer: torch.optim.Optimizer, tensors_dict: Dict[str, torch.Tensor]):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        if group["name"] not in tensors_dict:
            continue
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


def _replace(tensor: torch.Tensor, replace_mask: torch.Tensor, replace_tensor: torch.Tensor):
    tensor_clone = tensor.clone()
    tensor_clone[replace_mask] = replace_tensor
    return tensor_clone


def replace_tensors_to_optimizer(optimizer: torch.optim.Optimizer, tensors_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        if group["name"] not in tensors_dict:
            continue

        replace_mask, replace_tensor = tensors_dict[group["name"]]
        if replace_mask is None or replace_tensor is None:
            optimizable_tensors[group["name"]] = group["params"][0]
            continue

        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:

            stored_state["exp_avg"][replace_mask] = 0
            stored_state["exp_avg_sq"][replace_mask] = 0

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(_replace(group["params"][0], replace_mask, replace_tensor).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(_replace(group["params"][0], replace_mask, replace_tensor).requires_grad_(True))
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def mask_tensors_in_optimizer(optimizer: torch.optim.Optimizer, prune_mask: torch.Tensor, tensors_names: List[str]):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        if group["name"] not in tensors_names:
            continue
        mask = ~prune_mask
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:

            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
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
            scene_extent: float,
            densifier: AbstractDensifier,
            *args, **kwargs
    ):
        super().__init__(model, scene_extent, *args, **kwargs)
        self.densifier = densifier

    def add_points(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation):
        optimizable_tensors = cat_tensors_to_optimizer(self.optimizer, {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
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
        optimizable_tensors = mask_tensors_in_optimizer(self.optimizer, rm_mask, ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"])

        self.model.update_points_remove(
            removed_mask=rm_mask,
            xyz=optimizable_tensors["xyz"],
            features_dc=optimizable_tensors["f_dc"],
            features_rest=optimizable_tensors["f_rest"],
            opacity=optimizable_tensors["opacity"],
            scaling=optimizable_tensors["scaling"],
            rotation=optimizable_tensors["rotation"],
        )

    def replace_points(
        self,
        replace_xyz_mask, replace_xyz,
        replace_features_dc_mask, replace_features_dc,
        replace_features_rest_mask, replace_features_rest,
        replace_opacity_mask, replace_opacity,
        replace_scaling_mask, replace_scaling,
        replace_rotation_mask, replace_rotation,
    ):

        optimizable_tensors = replace_tensors_to_optimizer(self.optimizer, {
            "xyz": (replace_xyz_mask, replace_xyz),
            "f_dc": (replace_features_dc_mask, replace_features_dc),
            "f_rest": (replace_features_rest_mask, replace_features_rest),
            "opacity": (replace_opacity_mask, replace_opacity),
            "scaling": (replace_scaling_mask, replace_scaling),
            "rotation": (replace_rotation_mask, replace_rotation)})

        self.model.update_points_replace(
            xyz_mask=replace_xyz_mask, xyz=optimizable_tensors["xyz"],
            features_dc_mask=replace_features_dc_mask, features_dc=optimizable_tensors["f_dc"],
            features_rest_mask=replace_features_rest_mask, features_rest=optimizable_tensors["f_rest"],
            opacity_mask=replace_opacity_mask, opacity=optimizable_tensors["opacity"],
            scaling_mask=replace_scaling_mask, scaling=optimizable_tensors["scaling"],
            rotation_mask=replace_rotation_mask, rotation=optimizable_tensors["rotation"],
        )

    def densify_and_prune(self, loss, out, camera):
        instruct = self.densifier.densify_and_prune(loss, out, camera, self.curr_step)
        hook = False
        if instruct.replace_xyz_mask is not None  \
                or instruct.replace_features_dc_mask is not None \
                or instruct.replace_features_rest_mask is not None  \
                or instruct.replace_opacity_mask is not None \
                or instruct.replace_scaling_mask is not None  \
                or instruct.replace_rotation_mask is not None:
            if instruct.replace_xyz_mask is not None:
                assert instruct.replace_xyz is not None
            if instruct.replace_features_dc_mask is not None:
                assert instruct.replace_features_dc is not None
            if instruct.replace_features_rest_mask is not None:
                assert instruct.replace_features_rest is not None
            if instruct.replace_opacity_mask is not None:
                assert instruct.replace_opacity is not None
            if instruct.replace_scaling_mask is not None:
                assert instruct.replace_scaling is not None
            if instruct.replace_rotation_mask is not None:
                assert instruct.replace_rotation is not None
            self.replace_points(
                instruct.replace_xyz_mask, instruct.replace_xyz,
                instruct.replace_features_dc_mask, instruct.replace_features_dc,
                instruct.replace_features_rest_mask, instruct.replace_features_rest,
                instruct.replace_opacity_mask, instruct.replace_opacity,
                instruct.replace_scaling_mask, instruct.replace_scaling,
                instruct.replace_rotation_mask, instruct.replace_rotation,
            )
            hook = True
        if instruct.remove_mask is not None:
            self.remove_points(instruct.remove_mask)
            hook = True
        if instruct.new_xyz is not None:
            assert instruct.new_features_dc is not None
            assert instruct.new_features_rest is not None
            assert instruct.new_opacity is not None
            assert instruct.new_scaling is not None
            assert instruct.new_rotation is not None
            self.add_points(
                instruct.new_xyz,
                instruct.new_features_dc,
                instruct.new_features_rest,
                instruct.new_opacity,
                instruct.new_scaling,
                instruct.new_rotation)
            hook = True
        if hook:
            self.densifier.after_densify_and_prune_hook(loss, out, camera)
            torch.cuda.empty_cache()

    def before_optim_hook(self, loss, out, camera):
        with torch.no_grad():
            self.densify_and_prune(loss, out, camera)

    def from_densifier_constructor(
        densifier_constructor: Callable[..., AbstractDensifier],
            model: GaussianModel,
            scene_extent: float,
            *args,
            # copy from BaseTrainer
            lambda_dssim=0.2,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=30_000,
            feature_lr=0.0025,
            opacity_lr=0.025,
            scaling_lr=0.005,
            rotation_lr=0.001,
            mask_mode="none",
            bg_color=None,
            # copy from BaseTrainer
            **kwargs
    ) -> 'DensificationTrainer':
        densifier = densifier_constructor(model, scene_extent, *args, **kwargs)
        return DensificationTrainer(
            model=model,
            scene_extent=scene_extent,
            densifier=densifier,
            lambda_dssim=lambda_dssim,
            position_lr_init=position_lr_init,
            position_lr_final=position_lr_final,
            position_lr_delay_mult=position_lr_delay_mult,
            position_lr_max_steps=position_lr_max_steps,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr,
            mask_mode=mask_mode,
            bg_color=bg_color,
        )
