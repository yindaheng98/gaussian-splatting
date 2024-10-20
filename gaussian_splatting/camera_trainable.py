from .gaussian_model import GaussianModel
from .dataset import TrainableCameraDataset


class CameraTrainableGaussianModel(GaussianModel):
    def __init__(self, dataset: TrainableCameraDataset, *args, **kwargs):
        self.dataset = dataset
        super().__init__(*args, **kwargs)
        for k, v in self.dataset.get_params().items():
            self.register_parameter(k, v)

    def forward(self, idx: int):
        camera = self.dataset[idx]
        return super().forward(camera)

    def to(self, device):
        self.dataset.to(device)
        return super().to(device)
