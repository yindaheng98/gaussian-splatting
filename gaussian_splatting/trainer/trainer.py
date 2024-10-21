from abc import ABC, abstractmethod

import torch

from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.utils import l1_loss, ssim, get_expon_lr_func

from .densifier import Densifier


class AbstractTrainer(ABC):
    def __init__(self, model: GaussianModel):
        super().__init__()
        self.model = model
        self.model.active_sh_degree = 0

    def oneupSHdegree(self):
        if self.model.active_sh_degree < self.model.max_sh_degree:
            self.model.active_sh_degree += 1

    @abstractmethod
    def loss(self, out: dict, gt):
        pass

    def forward_backward(self, camera: Camera):
        out = self.model(camera)
        gt = camera.ground_truth_image
        loss = self.loss(out, gt)
        loss.backward()
        return loss, out, gt

    @abstractmethod
    def optim_step(self):
        pass

    def step(self, camera: Camera):
        loss, out, gt = self.forward_backward(camera)
        self.optim_step()
        return loss, out, gt


class BaseTrainer(AbstractTrainer):
    def __init__(
            self, model: GaussianModel,
            spatial_lr_scale: float,
            lambda_dssim=0.2,
            position_lr_init=0.00016,
            feature_lr=0.0025,
            opacity_lr=0.025,
            scaling_lr=0.005,
            rotation_lr=0.001,
            additional_optim_params=[],
    ):
        super().__init__(model)
        self.lambda_dssim = lambda_dssim
        params = [
            {'params': [model._xyz], 'lr': position_lr_init * spatial_lr_scale, "name": "xyz"},
            {'params': [model._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [model._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [model._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [model._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [model._rotation], 'lr': rotation_lr, "name": "rotation"}
        ] + additional_optim_params
        self.optimizer = torch.optim.Adam(params=params, lr=0.0, eps=1e-15)

    def loss(self, out: dict, gt):
        render = out["render"]
        Ll1 = l1_loss(render, gt)
        ssim_value = ssim(render, gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)
        return loss

    def optim_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


class LRScheduledTrainer(BaseTrainer):
    def __init__(
            self, model: GaussianModel,
            spatial_lr_scale: float,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=30_000,
            *args, **kwargs
    ):
        super().__init__(model, spatial_lr_scale=spatial_lr_scale, *args, position_lr_init=position_lr_init, **kwargs)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=position_lr_init*spatial_lr_scale,
            lr_final=position_lr_final*spatial_lr_scale,
            lr_delay_mult=position_lr_delay_mult,
            max_steps=position_lr_max_steps,
        )
        self.curr_step = 1

    def optim_step(self):
        self.update_learning_rate(self.curr_step)
        super().optim_step()
        self.curr_step += 1

    def update_learning_rate(self, step: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(step)
                param_group['lr'] = lr
                return lr


class DensificationTrainer(LRScheduledTrainer):
    def __init__(
            self, model: GaussianModel,
            densify_from_iter: int,
            densify_until_iter: int,
            densification_interval: int,
            opacity_reset_interval: int,
            scene_extent: float,
            densify_grad_threshold=0.0002,
            percent_dense=0.01,
            *args, **kwargs
    ):
        super().__init__(model, spatial_lr_scale=scene_extent, *args, **kwargs)
        self.scene_extent = scene_extent
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.percent_dense = percent_dense
        self.densifier = Densifier(model, self.optimizer)

    def step(self, camera):
        self.update_learning_rate(self.curr_step)
        loss, out, gt = self.forward_backward(camera)
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
        return loss, out, gt
