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

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from random import randint
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, kl_divergence, VGGPerceptualLoss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from gaussian_splatting.utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from gaussian_splatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn as nn
import numpy as np
import cv2

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from gaussian_splatting.data import DataLoader
from scene.cameras import Camera
import math

os.environ['TORCH_HOME'] = '/data/huangricong/commonModels/torch'

def training(dataset, opt, pipe, testing_iterations, saving_iterations, config):
    train_set = DataLoader(config, training=True).load_data()
    test_set = DataLoader(config, training=False).load_data()

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    # deform.load_weights(dataset.model_path)
    deform.train_setting(opt)

    # vgg_loss_func = VGGPerceptualLoss(resize=True).cuda()

    # if config.train_G:
    #     netG_input_nc = 6
    #     netG = 'local'
    #     output_nc = 3
    #     ngf = 16 # 32
    #     img_resolution = 512
    #     neural_rendering_resolution = 64
    #     n_downsample_global = int(math.log2(img_resolution / (neural_rendering_resolution / 2))) - 1
    #     n_blocks_global = 4
    #     n_local_enhancers = 1
    #     n_blocks_local = 3
    #     norm = 'instance'
    #     gpu_ids = []
    #     final = nn.Tanh()
    #     # final = nn.Sigmoid()
    #     netG = GeneratorModel(netG_input_nc, output_nc, ngf, netG,
    #                         n_downsample_global, n_blocks_global, n_local_enhancers, 
    #                         n_blocks_local, norm, gpu_ids, final=final, opt=opt)
        
    #     no_lsgan = False
    #     use_sigmoid = no_lsgan # false
    #     netD_input_nc = 3
    #     ndf = 64
    #     n_layers_D = 3
    #     norm = 'instance'
    #     num_D = 3
    #     no_ganFeat_loss = False
    #     netD = DiscriminatorModel(netD_input_nc, ndf, n_layers_D, norm, use_sigmoid, 
    #                                     num_D, not no_ganFeat_loss)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_epoch = 0
    epochs = 20
    progress_bar = tqdm(range(epochs), desc="Training progress")
    for epoch in range(epochs):
        # 设置学习率
        gaussians.update_learning_rate(int(epoch / epochs * 40000))
        deform.update_learning_rate(int(epoch / epochs * 40000))
        for iter, cam_info in enumerate(train_set):
            iteration = iter + 1
            iter_start.record()

            exp_coeffs = cam_info['base_code'][:, :, 80:144].squeeze(1).cuda() # (64)
            batch_size = exp_coeffs.shape[0]

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % (1000 // batch_size) == 0:
                gaussians.oneupSHdegree()

            # Render
            gt_image = torch.zeros((3, config.pred_img_size, config.pred_img_size))
            image_list = []
            visibility_filter_list = []
            radii_list = []
            viewspace_point_tensor_list = []
            for i in range(len(exp_coeffs)):
                # if epoch < opt.warm_up:
                if epoch < 1:
                    d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
                else:
                    N = gaussians.get_xyz.shape[0]
                    exp_coeffs_input = exp_coeffs[i:i+1].expand(N, -1)
                    d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), exp_coeffs_input)

                viewpoint_cam = Camera(colmap_id=0, 
                    R=(cam_info['cam_info']['batch_c2w_Rmats'][i].squeeze()).cpu().numpy(), 
                    T=(cam_info['cam_info']['batch_w2c_Tvecs'][i].squeeze()).cpu().numpy(),
                    # R=(cam_info['cam_info']['batch_c2w_Rmats'][i]), 
                    # T=(cam_info['cam_info']['batch_w2c_Tvecs'][i]),
                    FoVx=cam_info['cam_info']['FoVx'][i], FoVy=cam_info['cam_info']['FoVy'][i],
                    image=gt_image, # gt_alpha_mask=loaded_mask,
                    image_name='oo', uid=0,
                    # data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu',
                    exp_coeffs=exp_coeffs,
                    )
                render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                    "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
                image_list.append(image)
                visibility_filter_list.append(visibility_filter)
                radii_list.append(radii)
                viewspace_point_tensor_list.append(viewspace_point_tensor)

            image = torch.stack(image_list, dim=0)

            # Loss
            gt_image = cam_info['img_crop']
            gt_image[(cam_info['mask'] < 0.5).repeat(1, 3, 1, 1)] = 1. if dataset.white_background else 0.
            gt_image = gt_image.to(image.device)
            # cv2.imwrite(f"{args.model_path}/render.jpg", (image.permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1] * 255.).astype(np.uint8))
            # cv2.imwrite(f"{args.model_path}/gt.jpg", (gt_image[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1] * 255.).astype(np.uint8))

            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            mouth_area_mask = cam_info['mouth_area_mask'][0].to(image.device)
            image_mouth = image * mouth_area_mask
            gt_image_mouth = gt_image * mouth_area_mask
            Ll1_mouth = l1_loss(image_mouth, gt_image_mouth)
            loss_mouth = (1.0 - opt.lambda_dssim) * Ll1_mouth + opt.lambda_dssim * (1.0 - ssim(image_mouth, gt_image_mouth))
            
            loss = 1 * loss + 10. * loss_mouth
            # loss = 0.5 * loss + 5. * loss_mouth

            # loss += 1.0 * vgg_loss_func(image_mouth, gt_image_mouth)

            loss.backward()

            iter_end.record()

            # if dataset.load2gpu_on_the_fly:
            #     viewpoint_cam.load2device('cpu')

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Iteration": iteration * batch_size, "Loss": f"{ema_loss_for_log:.{7}f}", "Points": gaussians.get_xyz.shape[0]})
                    # progress_bar.update(10)

                # Keep track of max radii in image-space for pruning
                for visibility_filter, radii in zip(visibility_filter_list, radii_list):
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])
                
                # Densification
                if (epoch / epochs) < (opt.densify_until_iter / 40000):
                    for viewspace_point_tensor, visibility_filter in zip(viewspace_point_tensor_list, visibility_filter_list):
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if iteration % (opt.densification_interval // batch_size) == 0:
                        size_threshold = 20 #if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % (opt.opacity_reset_interval // batch_size) == 0 or (
                            dataset.white_background and iteration == (opt.densify_from_iter // batch_size)):
                        gaussians.reset_opacity()

                # Optimizer step
                gaussians.optimizer.step()
                # gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
        
        with torch.no_grad():
            # Log and save
            cur_psnr = training_report(tb_writer, epoch, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                    testing_iterations, scene, render, (pipe, background), deform,
                                    dataset.load2gpu_on_the_fly, dataset.is_6dof,
                                    train_set, test_set, config)
            if cur_psnr.item() > best_psnr:
                best_psnr = cur_psnr.item()
                best_epoch = epoch

            print("\n[Epoch {}] Saving Gaussians".format(epoch))
            scene.save(epoch, name="epoch")
            deform.save_weights(args.model_path, epoch, name="epoch")
        progress_bar.update(1)
    progress_bar.close()
    print("Best PSNR = {} in Epoch {}".format(best_psnr, best_epoch))


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


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, train_set=None, test_set=None, configs=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 
                        #    'cameras': scene.getTestCameras()
                        'cameras': test_set,
                        },
                           {'name': 'train',
                        #   'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                        #               range(5, 30, 5)]
                            'cameras': train_set,
                                        }
                        )

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            images = torch.tensor([], device="cuda")
            gts = torch.tensor([], device="cuda")
            for idx, cam_info in enumerate(config['cameras']):
                if config['name'] == 'train' and idx >= 50:
                    break
                if load2gpu_on_the_fly:
                    viewpoint.load2device()
                xyz = scene.gaussians.get_xyz
                # fid = viewpoint.fid
                # time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                # exp_coeffs = viewpoint.exp_coeffs
                exp_coeffs = cam_info['base_code'][:, :, 80:144].squeeze(1).cuda()
                for i in range(len(exp_coeffs)):
                    exp_coeffs_input = exp_coeffs[i:i+1].expand(xyz.shape[0], -1)

                    gt_image = cam_info['img_crop'][i:i+1]
                    gt_image[(cam_info['mask'][i:i+1] < 0.5).repeat(1, 3, 1, 1)] = 1. if configs.white_background else 0.
                    viewpoint = Camera(colmap_id=0, R=(cam_info['cam_info']['batch_c2w_Rmats'][i].squeeze()).cpu().numpy(), T=(cam_info['cam_info']['batch_w2c_Tvecs'][i].squeeze()).cpu().numpy(),
                        FoVx=cam_info['cam_info']['FoVx'][i], FoVy=cam_info['cam_info']['FoVy'][i],
                        image=gt_image.squeeze(), # gt_alpha_mask=loaded_mask,
                        image_name='oo', uid=0,
                        # data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu',
                        exp_coeffs=exp_coeffs,)
            
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), exp_coeffs_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                if load2gpu_on_the_fly:
                    viewpoint.load2device('cpu')
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                            image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                gt_image[None], global_step=iteration)

            l1_test = l1_loss(images, gts)
            psnr_test = psnr(images, gts).mean()
            if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                test_psnr = psnr_test
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--txt_test_path", type=str, required=True)
    parser.add_argument("--pred_img_size", type=int, required=True)
    parser.add_argument("--sample_size", default=[512, 512])
    parser.add_argument("--batch_size", default=1, type=int)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)

    # All done
    print("\nTraining complete.")
