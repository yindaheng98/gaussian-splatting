from gaussian_splatting import Camera
from torch.utils.data import Dataset


class CameraDataset(Dataset):

    def __getitem__(self, idx) -> Camera:
        pass
