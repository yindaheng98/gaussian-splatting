import torch

from gaussian_splatting.camera import Camera


def exposure_postprocess(camera: Camera, x: torch.Tensor):
    exposure = camera.custom_data['exposures']
    return torch.matmul(x.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]
