from abc import abstractmethod
from gaussian_splatting import Camera
from torch.utils.data import Dataset


class CameraDataset(Dataset):

    @abstractmethod
    def to(self, device) -> 'CameraDataset':
        return self

    @abstractmethod
    def __getitem__(self, idx) -> Camera:
        pass
