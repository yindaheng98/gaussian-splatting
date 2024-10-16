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

import os
import torch
from random import randint
from gaussian_splatting.dataset.colmap.dataset import ColmapCameraDataset
from gaussian_splatting.trainer.colmap import ColmapTrainer
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from gaussian_splatting import GaussianModel as NewGaussianModel, Camera
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def compute_difference(gaussians: GaussianModel, new_gaussians: NewGaussianModel):
    with torch.no_grad():
        diff_xyz = torch.abs(gaussians._xyz - new_gaussians._xyz).max().item()
        diff_features_dc = torch.abs(gaussians._features_dc - new_gaussians._features_dc).max().item()
        diff_features_rest = torch.abs(gaussians._features_rest - new_gaussians._features_rest).max().item()
        diff_scaling = torch.abs(gaussians._scaling - new_gaussians._scaling).max().item()
        diff_rotation = torch.abs(gaussians._rotation - new_gaussians._rotation).max().item()
        diff_opacity = torch.abs(gaussians._opacity - new_gaussians._opacity).max().item()
        print("Differences params: ", diff_xyz, diff_features_dc, diff_features_rest, diff_scaling, diff_rotation, diff_opacity)


def compute_difference_out(out, new_out):
    with torch.no_grad():
        diff_render = torch.abs(out["render"] - new_out["render"]).max().item()
        diff_viewspace_points = torch.abs(out["viewspace_points"] - new_out["viewspace_points"]).max().item()
        diff_visibility_filter = torch.abs(out["visibility_filter"] - new_out["visibility_filter"]).max().item()
        diff_radii = torch.abs(out["radii"] - new_out["radii"]).max().item()
        diff_depth = torch.abs(out["depth"] - new_out["depth"]).max().item()
        print("Differences out: ", diff_render, diff_viewspace_points, diff_visibility_filter, diff_radii, diff_depth)


def compute_difference_grad(gaussians: GaussianModel, new_gaussians: NewGaussianModel):
    diff_xyz = torch.abs(gaussians._xyz.grad - new_gaussians._xyz.grad).max().item()
    diff_features_dc = torch.abs(gaussians._features_dc.grad - new_gaussians._features_dc.grad).max().item()
    diff_features_rest = torch.abs(gaussians._features_rest.grad - new_gaussians._features_rest.grad).max().item()
    diff_scaling = torch.abs(gaussians._scaling.grad - new_gaussians._scaling.grad).max().item()
    diff_rotation = torch.abs(gaussians._rotation.grad - new_gaussians._rotation.grad).max().item()
    diff_opacity = torch.abs(gaussians._opacity.grad - new_gaussians._opacity.grad).max().item()
    print("Differences grads: ", diff_xyz, diff_features_dc, diff_features_rest, diff_scaling, diff_rotation, diff_opacity)


def sync_grad(gaussians: GaussianModel, new_gaussians: NewGaussianModel):
    new_gaussians._xyz.grad[:] = gaussians._xyz.grad
    new_gaussians._features_dc.grad[:] = gaussians._features_dc.grad
    new_gaussians._features_rest.grad[:] = gaussians._features_rest.grad
    new_gaussians._scaling.grad[:] = gaussians._scaling.grad
    new_gaussians._rotation.grad[:] = gaussians._rotation.grad
    new_gaussians._opacity.grad[:] = gaussians._opacity.grad


def compute_difference_optim(optim: torch.optim.Optimizer, new_optim: torch.optim.Optimizer):
    record = ''
    for param_group in optim.param_groups:
        for new_param_group in new_optim.param_groups:
            if param_group["name"] == new_param_group["name"]:
                diff_lr = param_group["lr"] - new_param_group["lr"]
                record += f"{param_group['name']} lr: {diff_lr} "
    print("Differences lr: ", record)


def compute_difference_densification_stats(gaussians: GaussianModel, new_gaussians: ColmapTrainer):
    densifier = new_gaussians.densifier
    with torch.no_grad():
        diff_max_radii2D = torch.abs(gaussians.max_radii2D - densifier.max_radii2D).max().item()
        diff_xyz_gradient_accum = torch.abs(gaussians.xyz_gradient_accum - densifier.xyz_gradient_accum).max().item()
        diff_denom = torch.abs(gaussians.denom - densifier.denom).max().item()
        print("Differences stats: ", diff_max_radii2D, diff_xyz_gradient_accum, diff_denom)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    new_gaussians = NewGaussianModel(dataset.sh_degree).to("cuda")
    new_dataset = ColmapCameraDataset(dataset.source_path)
    trainer = ColmapTrainer(
        new_gaussians,
        os.path.join(dataset.source_path, "sparse/0/points3D.bin"),
        dataset=new_dataset,
        densify_from_iter=opt.densify_from_iter,
        densify_until_iter=opt.densify_until_iter,
        densification_interval=opt.densification_interval,
    )
    compute_difference(gaussians, new_gaussians)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        trainer.update_learning_rate(iteration)
        compute_difference_optim(gaussians.optimizer, trainer.optimizer)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            trainer.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        compute_difference(gaussians, new_gaussians)
        camera = Camera(
            image_height=viewpoint_cam.image_height,
            image_width=viewpoint_cam.image_width,
            FoVx=viewpoint_cam.FoVx,
            FoVy=viewpoint_cam.FoVy,
            world_view_transform=viewpoint_cam.world_view_transform,
            full_proj_transform=viewpoint_cam.full_proj_transform,
            camera_center=viewpoint_cam.camera_center,
            bg_color=bg,
            ground_truth_image=gt_image,
        )
        new_loss, new_out, new_gt = trainer.forward_backward(camera)
        compute_difference_grad(gaussians, new_gaussians)
        sync_grad(gaussians, new_gaussians)  # we can verify that the wrong gradients are the cause of difference
        compute_difference_grad(gaussians, new_gaussians)

        iter_end.record()

        with torch.no_grad():
            print("loss diff", loss.item() - new_loss.item())
            compute_difference_out(render_pkg, new_out)
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            compute_difference_densification_stats(gaussians, trainer)
            new_out["viewspace_points"].grad[:] = viewspace_point_tensor.grad[:]  # sync grad
            trainer.update_densification_stats(new_out)
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            compute_difference_densification_stats(gaussians, trainer)

            # Optimizer step
            if iteration < opt.iterations:
                # gaussians.exposure_optimizer.step()
                # gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                trainer.optim_step()
                compute_difference(gaussians, new_gaussians)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
