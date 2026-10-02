import argparse
import cv2
import glob
import os
import numpy as np
import torch
import sys
import tifffile


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", 'Depth-Anything-V2'))
    from depth_anything_v2.dpt import DepthAnythingV2
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('-s', '--source', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    image_dir = os.path.join(args.source, "images")
    depth_dir = os.path.join(args.source, "depths")
    depth_mask_dir = os.path.join(args.source, "depth_masks")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_mask_dir, exist_ok=True)
    filenames = [f for f in glob.glob(os.path.join(image_dir, '**/*'), recursive=True) if os.path.isfile(f)]

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        raw_image = cv2.imread(filename)

        depth = depth_anything.infer_image(raw_image, args.input_size)

        rel = os.path.relpath(filename, image_dir)
        depth_tiff = os.path.join(depth_dir, rel + '.tiff')
        depth_png = os.path.join(depth_dir, rel + '.png')
        mask_tiff = os.path.join(depth_mask_dir, rel + '.tiff')
        mask_png = os.path.join(depth_mask_dir, rel + '.png')
        os.makedirs(os.path.dirname(depth_tiff), exist_ok=True)
        os.makedirs(os.path.dirname(mask_tiff), exist_ok=True)

        tifffile.imwrite(depth_tiff, depth)
        tifffile.imwrite(mask_tiff, np.ones_like(depth))

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        cv2.imwrite(depth_png, depth)
        cv2.imwrite(mask_png, np.full(depth.shape, 255, dtype=np.uint8))
