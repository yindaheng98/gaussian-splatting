from typing import Callable
from .gaussian_model import GaussianModel
from .dataset import CameraDataset, FixedTrainableCameraDataset, TrainableCameraDataset
from .dataset.colmap import ColmapCameraDataset, ColmapTrainableCameraDataset, colmap_init
from .trainer import *
from .trainer.extensions import ScaleRegularizeTrainerWrapper


def prepare_dataset(source: str, device: str, trainable_camera: bool = False, load_camera: str = None, load_mask=True, load_depth=True) -> CameraDataset:
    if trainable_camera:
        dataset = (
            TrainableCameraDataset.from_json(load_camera, load_mask=load_mask, load_depth=load_depth)
            if load_camera else
            ColmapTrainableCameraDataset(source, load_mask=load_mask, load_depth=load_depth)
        ).to(device)
    else:
        dataset = (
            FixedTrainableCameraDataset(load_camera, load_mask=load_mask, load_depth=load_depth)
            if load_camera else
            ColmapCameraDataset(source, load_mask=load_mask, load_depth=load_depth)
        ).to(device)
    return dataset


backends = ["inria", "gsplat", "gsplat-2dgs"]


def get_gaussian_model_class(backend: str, trainable_camera: bool = False) -> Callable[[int], GaussianModel]:
    match backend:
        case "inria":
            from .gaussian_model import GaussianModel
            from .camera_trainable import CameraTrainableGaussianModel
            return GaussianModel if not trainable_camera else CameraTrainableGaussianModel
        case "gsplat":
            from .models import GsplatGaussianModel, CameraTrainableGsplatGaussianModel
            return GsplatGaussianModel if not trainable_camera else CameraTrainableGsplatGaussianModel
        case "gsplat-2dgs":
            from .models import Gsplat2DGSGaussianModel, CameraTrainableGsplat2DGSGaussianModel
            return Gsplat2DGSGaussianModel if not trainable_camera else CameraTrainableGsplat2DGSGaussianModel
        case _:
            raise ValueError(f"Unknown backend: {backend}")


def prepare_gaussians(sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None, backend: str = "inria") -> GaussianModel:
    gaussians = get_gaussian_model_class(backend, trainable_camera=trainable_camera)(sh_degree).to(device)
    gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    return gaussians


basemodes = {
    "base": Trainer,
    "densify": OpacityResetDensificationTrainer,
    "camera": CameraTrainer,
    "camera-densify": OpacityResetDensificationCameraTrainer,
    "nodepth-base": BaseTrainer,
    "nodepth-densify": BaseOpacityResetDensificationTrainer,
    "nodepth-camera": BaseCameraTrainer,
    "nodepth-camera-densify": BaseOpacityResetDensificationCameraTrainer,
}
shliftmodes = {
    "base": SHLiftTrainer,
    "densify": SHLiftOpacityResetDensificationTrainer,
    "camera": SHLiftCameraTrainer,
    "camera-densify": SHLiftOpacityResetDensificationCameraTrainer,
    "nodepth-base": BaseSHLiftTrainer,
    "nodepth-densify": BaseSHLiftOpacityResetDensificationTrainer,
    "nodepth-camera": BaseSHLiftCameraTrainer,
    "nodepth-camera-densify": BaseSHLiftOpacityResetDensificationCameraTrainer,
}


def prepare_trainer(gaussians: GaussianModel, dataset: CameraDataset, mode: str, load_ply: str = None, with_scale_reg=False, configs={}) -> AbstractTrainer:
    modes = shliftmodes if load_ply else basemodes
    constructor = modes[mode]
    if with_scale_reg:
        constructor = lambda *args, **configs: ScaleRegularizeTrainerWrapper(modes[mode], *args, **configs)
    trainer = constructor(gaussians, dataset, **configs)
    return trainer
