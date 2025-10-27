from typing import Callable, Dict

import torch

from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.utils import l1_loss, ssim
from gaussian_splatting.utils.schedular import get_expon_lr_func

from .abc import AbstractTrainer


class BaseTrainer(AbstractTrainer):
    def __init__(
            self, model: GaussianModel,
            scene_extent: float,
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
            # "none"=do not use mask
            # "ignore"=loss of the masked area will be set to 0
            # "bg_color"=fill the masked area of ground truth with the bg_color for rendering
            bg_color=None,
            # None=do not change bg_color
            # "random"=set bg_color to random color
            # tuple(float, float, float)=set bg_color to the given color
    ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        assert mask_mode in ["none", "ignore", "bg_color"], f"Unknown mask policy: {mask_mode}"
        assert bg_color is None or bg_color == "random" or (
            isinstance(bg_color, tuple) and len(bg_color) == 3 and all(isinstance(c, float) for c in bg_color)
        ), f"bg_color must be 'random' or a RGB value tuple(float, float, float), got {bg_color}"
        self.mask_mode = mask_mode
        self.bg_color = bg_color
        params = [
            {'params': [model._xyz], 'lr': position_lr_init * scene_extent, "name": "xyz"},
            {'params': [model._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [model._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [model._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [model._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [model._rotation], 'lr': rotation_lr, "name": "rotation"}
        ]
        optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        schedulers = {
            "xyz": get_expon_lr_func(
                lr_init=position_lr_init*scene_extent,
                lr_final=position_lr_final*scene_extent,
                lr_delay_mult=position_lr_delay_mult,
                max_steps=position_lr_max_steps,
            )
        }
        self._model = model
        self._optimizer = optimizer
        self._schedulers = schedulers
        self._curr_step = 0

    @property
    def curr_step(self) -> int:
        return self._curr_step

    @curr_step.setter
    def curr_step(self, v):
        self._curr_step = v

    @property
    def model(self) -> GaussianModel:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def schedulers(self) -> Dict[str, Callable[[int], float]]:
        return self._schedulers

    def preprocess(self, camera: Camera) -> Camera:
        if self.bg_color == "random":
            camera = camera._replace(bg_color=torch.rand_like(camera.bg_color))
        elif isinstance(self.bg_color, tuple):
            camera = camera._replace(bg_color=torch.tensor(self.bg_color, device=camera.bg_color.device, dtype=camera.bg_color.dtype))
        elif self.bg_color is None:
            pass
        else:
            raise ValueError(f"bg_color must be 'random' or a tuple(int, int, int), got {self.bg_color}")
        return camera

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        render = out["render"]
        gt = camera.ground_truth_image
        mask = camera.ground_truth_image_mask
        match self.mask_mode:
            case "none":
                pass
            case "ignore":
                assert mask is not None, "Mask is required for 'ignore' mask policy"
                render = render * mask.unsqueeze(0)
                gt = gt * mask.unsqueeze(0)
            case "bg_color":
                assert mask is not None, "Mask is required for 'bg_color' mask policy"
                # bg_color after postprocess
                bg_color = camera.postprocess(camera, camera.bg_color.unsqueeze(-1).unsqueeze(-1)).clamp(0.0, 1.0)
                gt = gt * mask.unsqueeze(0) + (1 - mask.unsqueeze(0)) * bg_color
            case _:
                raise ValueError(f"Unknown mask policy: {self.mask_mode}")
        Ll1 = l1_loss(render, gt)
        ssim_value = ssim(render, gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)
        return loss
