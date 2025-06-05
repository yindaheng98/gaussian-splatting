from .gaussian_model import GaussianModel
from .camera_trainable import CameraTrainableGaussianModel
from .dataset import CameraDataset, JSONCameraDataset, TrainableCameraDataset
from .dataset.colmap import ColmapCameraDataset, ColmapTrainableCameraDataset, colmap_init
from .trainer import *
from .trainer.extensions import ScaleRegularizeTrainerWrapper


def prepare_dataset(source: str, device: str, trainable_camera: bool = False, load_camera: str = None, load_depth=False) -> CameraDataset:
    if trainable_camera:
        dataset = (TrainableCameraDataset.from_json(load_camera, load_depth=load_depth) if load_camera else ColmapTrainableCameraDataset(source, load_depth=load_depth)).to(device)
    else:
        dataset = (JSONCameraDataset(load_camera, load_depth=load_depth) if load_camera else ColmapCameraDataset(source, load_depth=load_depth)).to(device)
    return dataset


def prepare_gaussians(sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None) -> GaussianModel:
    if trainable_camera:
        gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
        gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
    else:
        gaussians = GaussianModel(sh_degree).to(device)
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


def prepare_trainer(gaussians: GaussianModel, dataset: CameraDataset, mode: str, trainable_camera: bool = False, load_ply: str = None, with_scale_reg=False, configs={}) -> AbstractTrainer:
    modes = shliftmodes if load_ply else basemodes
    constructor = modes[mode]
    if with_scale_reg:
        constructor = lambda *args, **kwargs: ScaleRegularizeTrainerWrapper(modes[mode], *args, **kwargs)
    if trainable_camera:
        trainer = constructor(
            gaussians,
            scene_extent=dataset.scene_extent(),
            dataset=dataset,
            **configs
        )
    else:
        trainer = constructor(
            gaussians,
            scene_extent=dataset.scene_extent(),
            **configs
        )
    return trainer
