from typing import Dict, List
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
            scene_extent: float,
            densifier: AbstractDensifier,
            *args, **kwargs
    ):
        super().__init__(model, scene_extent, *args, **kwargs)
        self.densifier = densifier

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

    def densify_and_prune(self, loss, out, camera):
        instruct = self.densifier.densify_and_prune(loss, out, camera, self.curr_step)
        hook = False
        if instruct.remove_mask is not None:
            self.remove_points(instruct.remove_mask)
            hook = True
        if instruct.new_xyz is not None:
            assert instruct.new_features_dc is not None
            assert instruct.new_features_rest is not None
            assert instruct.new_opacities is not None
            assert instruct.new_scaling is not None
            assert instruct.new_rotation is not None
            self.add_points(
                instruct.new_xyz,
                instruct.new_features_dc,
                instruct.new_features_rest,
                instruct.new_opacities,
                instruct.new_scaling,
                instruct.new_rotation)
            hook = True
        if hook:
            self.densifier.after_densify_and_prune_hook(loss, out, camera)
            torch.cuda.empty_cache()

    def before_optim_hook(self, loss, out, camera):
        with torch.no_grad():
            self.densify_and_prune(loss, out, camera)
