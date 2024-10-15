#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel as OldGaussianModel
from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.dataset import ColmapCameraDataset


def render_set(model_path, name, iteration, views, gaussians, new_gaussians, new_dataset, pipeline, background, train_test_exp):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        camera = Camera(
            image_height=view.image_height,
            image_width=view.image_width,
            FoVx=view.FoVx,
            FoVy=view.FoVy,
            world_view_transform=view.world_view_transform,
            full_proj_transform=view.full_proj_transform,
            camera_center=view.camera_center,
            bg_color=background
        )
        out = new_gaussians(camera)
        difference = torch.abs(out["render"] - rendering)
        print("difference", difference.sum())
        new_view = new_dataset[idx]
        new_gt = new_view.ground_truth_image
        difference = torch.abs(new_gt - gt)
        print("difference", difference.sum())
        out = new_gaussians(new_view)
        difference = torch.abs(out["render"] - rendering)
        print("difference", difference.sum())


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = OldGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        new_gaussians = GaussianModel(dataset.sh_degree)
        new_gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))
        new_dataset = ColmapCameraDataset(dataset.source_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, new_gaussians, new_dataset, pipeline, background, dataset.train_test_exp)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, new_gaussians, new_dataset, pipeline, background, dataset.train_test_exp)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)