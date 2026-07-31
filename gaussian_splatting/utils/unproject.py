import torch


def unproject(depth, K):
    height, width = depth.shape
    K = K.to(device=depth.device, dtype=depth.dtype)
    uv = torch.ones((height, width, 3), dtype=depth.dtype, device=depth.device)
    uv[..., 0] = torch.arange(0, width, dtype=depth.dtype).unsqueeze(0).expand(height, -1)
    uv[..., 1] = torch.arange(0, height, dtype=depth.dtype).unsqueeze(1).expand(-1, width)
    xyz = torch.inverse(K) @ uv.reshape(-1, 3).T * depth.reshape(-1)
    return xyz.T.reshape(*uv.shape)
