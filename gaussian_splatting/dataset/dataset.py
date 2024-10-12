from gaussian_splatting import Camera
from torch.utils.data import Dataset


class CameraDataset(Dataset):

    def to(self, device) -> 'CameraDataset':
        raise NotImplementedError("Subclasses of CameraDataset should implement to(device).")

    def __getitem__(self, idx) -> Camera:
        raise NotImplementedError("Subclasses of CameraDataset should implement __getitem__.")
