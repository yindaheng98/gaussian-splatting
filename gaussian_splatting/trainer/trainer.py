from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.utils import l1_loss, ssim

from .densifier import Densifier


class AbstractTrainer(ABC):

    @abstractmethod
    def step(self, camera: Camera):
        pass

    def train_epoch(self, cameras: CameraDataset, current_epoch: int, total_epochs: int):
        pbar = tqdm(cameras, desc=f"Epoch {current_epoch}/{total_epochs}")
        for camera in pbar:
            loss = self.train_step(camera)
            if loss is not None:
                pbar.set_postfix(loss=loss.item())

    def train(self, cameras: CameraDataset, num_epochs):
        for epoch in range(num_epochs):
            self.train_epoch(cameras, epoch, num_epochs)


class Trainer(AbstractTrainer):
    def __init__(
            self, model: GaussianModel,
            spatial_lr_scale: float,
            position_lr_init=0.00016,
            feature_lr=0.0025,
            opacity_lr=0.025,
            scaling_lr=0.005,
            rotation_lr=0.001,
            lambda_dssim=0.2,
    ):
        super().__init__()
        self.model = model
        l = [
            {'params': [model._xyz], 'lr': position_lr_init * spatial_lr_scale, "name": "xyz"},
            {'params': [model._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [model._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [model._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [model._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [model._rotation], 'lr': rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.lambda_dssim = lambda_dssim

    def step(self, camera: Camera):
        out = self.model(camera)
        render = out["render"]
        gt = camera.ground_truth_image
        Ll1 = l1_loss(render, gt)
        ssim_value = ssim(render, gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        return loss, out, gt


class DensificationTrainer(Trainer):
    def __init__(
            self, model: GaussianModel,
            densify_from_iter: int,
            densify_until_iter: int,
            densification_interval: int,
            grad_threshold=0.1,
            percent_dense=0.1,
            scene_extent=1.0,
            *args, **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.grad_threshold = grad_threshold
        self.percent_dense = percent_dense
        self.scene_extent = scene_extent
        self.curr_step = 0
        self.densifier = Densifier(model)

    def step(self, camera):
        self.curr_step += 1
        loss, out, gt = super().step(camera)
        render, viewspace_points, visibility_filter, radii = out["render"], out["viewspace_points"], out["visibility_filter"], out["radii"]
        if self.curr_step < self.densify_until_iter:
            self.densifier.update_densification_stats(radii, viewspace_points, visibility_filter)
        return loss, out, gt
