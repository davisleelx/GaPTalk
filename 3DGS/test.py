import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from scene import Scene, DeformModel
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
from gaussian_splatting.data import DataLoader
from scene.cameras import Camera
import cv2
import torch.nn.functional as F
from NetWorks import networks_modified as networks
from gaussian_splatting.models.metaportrait.archs.vgfpgan_gfpganv1_clean_unetwindow_arch import GFPGANv1Clean_UnetWindow
from omegaconf import OmegaConf
import time
from Utils.load_model import load_state_dict

class FittingImage(object):
    def __init__(self, dataset, iteration=-1) -> None:
        super().__init__()

        self.gaussians = GaussianModel(dataset.sh_degree)
        name = "epoch"
        self.scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False, name=name)
        self.deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        self.deform.load_weights(dataset.model_path, name=name)
        self.dataset = dataset

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        netG = networks.define_G(**dataset.netG)
        netG.load_state_dict(torch.load(dataset.netG.model_path, map_location="cpu")['net'])
        self.netG = netG.cuda()

        # netG = GFPGANv1Clean_UnetWindow(**dataset.netG.network_g)
        # load_state_dict(netG, torch.load(dataset.netG.model_path, map_location=torch.device("cpu"))['net'])
        # self.netG = netG.cuda()

    def rescale_mask(self, input_mask, transform_params: list, img_ori, use_bg=False):
        """
        Uncrops and rescales (i.e., resizes) the given scaled and cropped mask back to the
        resolution of the original image using the given transformation parameters.
        input_mask: (B, 3, H, W)
        """
        target_size = 1024.
        rescale_factor = 300
        center_crop_size = 700
        output_size = 512

        input_mask = input_mask.cpu()
        device = input_mask.device
        B = len(input_mask)
        original_image_width, original_image_height = transform_params[:, 0].to(torch.int32), transform_params[:, 1].to(torch.int32)
        s = transform_params[:, 2]
        t = transform_params[:, 3:] # (2)

        scaled_image_w = (original_image_width * s).to(torch.int32)
        scaled_image_h = (original_image_height * s).to(torch.int32)
        left = (scaled_image_w/2 - target_size/2 + (t[:, 0] - original_image_width/2)*s).to(torch.int32)
        up = (scaled_image_h/2 - target_size/2 + (original_image_height/2 - t[:, 1])*s).to(torch.int32)

        # Parse transform params.
        mask_scaled_list = []
        for i in range(B):
            mask_scaled_0 = torch.ones([3, int(target_size), int(target_size)]).to(device)
            # if not use_bg:
            #     img = F.interpolate(img_ori_tensor[[i]], (scaled_image_h, scaled_image_w), mode='bilinear')[0]

            x = F.interpolate(input_mask[[i]], (center_crop_size, center_crop_size), mode='bilinear')
            mask_scaled_0[:, 512-350:512+350, 512-350:512+350] = x
            x = mask_scaled_0

            if left[i] < 0:
                left_scale = 0
                right_scale = int(left_scale + min(target_size + left[i], scaled_image_w))

                left_target = -left[i]
                right_target = int(left_target + min(target_size - left_target, scaled_image_w))

            else:
                left_scale = left[i]
                right_scale = int(left_scale + min(target_size, scaled_image_w - left[i]))

                left_target = 0
                right_target = int(left_target + min(target_size, scaled_image_w - left[i]))

            if up[i] < 0:
                up_scale = 0
                below_sclae = int(up_scale + min(target_size + up[i], scaled_image_h))

                up_target = -up[i]
                below_target = int(up_target + min(target_size - up_target, scaled_image_h))

            else:
                up_scale = up[i]
                below_sclae = int(up_scale + min(target_size, scaled_image_h - up[i]))

                up_target = 0
                below_target = int(up_target + min(target_size, scaled_image_h - up[i]))

            # if not use_bg:
            #     mask_scaled = img
            # else:
            #     mask_scaled = torch.ones([3, scaled_image_h[i], scaled_image_w[i]]).to(device)
            mask_scaled = torch.ones([3, scaled_image_h[i], scaled_image_w[i]]).to(device)
            mask_scaled[:, up_scale:below_sclae, left_scale:right_scale] = \
                x[:, up_target:below_target, left_target:right_target]
            # Rescale the uncropped mask back to the resolution of the original image.
            uncropped_and_rescaled_mask = F.interpolate(mask_scaled.unsqueeze(0), (original_image_height[i], original_image_width[i]), mode='bilinear')
            
            uncropped_and_rescaled_mask = uncropped_and_rescaled_mask.cpu().numpy()
            mask = (uncropped_and_rescaled_mask[0, 0]!=1).astype(np.float32)
            # soft_mask = torch.from_numpy(cv2.GaussianBlur(mask, (9, 9), 0)).cuda()
            soft_mask = cv2.erode(mask, np.ones((3, 3), np.uint8))

            # uncropped_and_rescaled_mask[uncropped_and_rescaled_mask==1] = img_ori_tensor[[i]][uncropped_and_rescaled_mask==1]
            uncropped_and_rescaled_mask = soft_mask * uncropped_and_rescaled_mask + (1 - soft_mask) * img_ori
            # uncropped_and_rescaled_mask[uncropped_and_rescaled_mask==0] = img_ori_tensor[[i]][uncropped_and_rescaled_mask==0]
            # uncropped_and_rescaled_mask[uncropped_and_rescaled_mask>=0.8] = img_ori_tensor[[i]][uncropped_and_rescaled_mask>=0.8]
            
            mask_scaled_list.append(uncropped_and_rescaled_mask)
        
        uncropped_and_rescaled_mask = np.concatenate(mask_scaled_list, axis=0)

        return uncropped_and_rescaled_mask
    
    def perform_fitting(self, exp_coeffs, cam_info, 
            rest_img_crop,
            pipeline,
            trans_params, img_ori_tensor,
            face_mask=None):
        L = exp_coeffs.shape[0]
        results = {'render': []}
        for i in range(L):
            gt_image = torch.zeros((3, self.dataset.pred_img_size, self.dataset.pred_img_size))
            view = Camera(colmap_id=0, R=(cam_info['cam_info']['batch_c2w_Rmats'].squeeze(0)[i]).cpu().numpy(), T=(cam_info['cam_info']['batch_w2c_Tvecs'].squeeze(0, 3)[i]).cpu().numpy(),
                    FoVx=cam_info['cam_info']['FoVx'], FoVy=cam_info['cam_info']['FoVy'],
                    image=gt_image, # gt_alpha_mask=loaded_mask,
                    image_name='oo', uid=0,
                    # data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu',
                    exp_coeffs=exp_coeffs[i],)
        
            if self.dataset.load2gpu_on_the_fly:
                view.load2device()
            xyz = self.gaussians.get_xyz

            exp_coeffs = view.exp_coeffs
            exp_coeffs_input = exp_coeffs.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = self.deform.step(xyz.detach(), exp_coeffs_input)

            results['render'].append(render(view, self.gaussians, pipeline, self.background, d_xyz, d_rotation, d_scaling, self.dataset.is_6dof)['render'])
        results['render'] = torch.stack(results['render'], dim=0)

        x = rest_img_crop
        y = results["render"] * 2. - 1.
        # y = self.rescale_mask(results["render"], trans_params, np.zeros_like(img_ori_tensor), use_bg=False)
        
        # result_img = self.netG(torch.cat([x, y], dim=1))
        # start = time.time()
        result_img = self.netG(x, y)
        # result_img = self.netG(torch.from_numpy(y).unsqueeze(0)).squeeze(0)
        # end = time.time()
        # self.all_time += end - start
        result_img = (result_img + 1.) / 2.
        
        # x[y != -1] = y[y != -1]
        # x = x * (1 - face_mask) + y * face_mask
        # result_img = (x + 1.) / 2.

        coarse_fg_rgb = self.rescale_mask(result_img, trans_params, img_ori_tensor, use_bg=False)
        # coarse_fg_rgb = result_img.cpu().numpy()
        # coarse_fg_rgb_1 = self.rescale_mask(results["render"].unsqueeze(0), trans_params, img_ori_tensor, use_bg=False)
        coarse_fg_rgb_1 = self.rescale_mask(results["render"], trans_params, np.zeros_like(img_ori_tensor), use_bg=False)

        coarse_fg_rgb_list = []
        coarse_fg_rgb_list_1 = []
        for i in range(len(coarse_fg_rgb)):
            coarse_fg_rgb_list.append((coarse_fg_rgb[i].transpose(1, 2, 0) * 255.).clip(0, 255).astype(np.uint8))
            coarse_fg_rgb_list_1.append((coarse_fg_rgb_1[i].transpose(1, 2, 0) * 255.).clip(0, 255).astype(np.uint8))

        imgs = {
            'result_img': coarse_fg_rgb_list,
            # 'implict_img': [(results["render"].permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)],
            'implict_img': coarse_fg_rgb_list_1,
                }
        return imgs

    def render_set(self, name, views, pipeline):
        render_path = os.path.join(self.dataset.model_path, name, "ours_{}".format(self.scene.loaded_iter), "renders")
        # gts_path = os.path.join(self.dataset.model_path, name, "ours_{}".format(self.scene.loaded_iter), "gt")

        makedirs(render_path, exist_ok=True)
        # makedirs(gts_path, exist_ok=True)
        self.all_time = 0
        for idx, cam_info in enumerate(tqdm(views, desc="Rendering progress")):
            exp_coeffs = cam_info['base_code'][:, :, 80:144].squeeze()
            results = self.perform_fitting(exp_coeffs, cam_info, cam_info['img_crop'].cuda(),
                                           pipeline, cam_info['trans_params'], cam_info['img_ori'],)
            rendering = results["render"]
            
            # gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # print("FPS: ", len(views) / self.all_time)

def interpolate_time(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_view(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_view_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    frame = 150
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    render_poses = torch.stack(render_wander_path(view), 0)
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
    #                            0)

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = view.fid

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        # acc = results["acc"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))
        # torchvision.utils.save_image(acc, os.path.join(acc_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]],
                               0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_poses(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, timer):
    render_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_pose_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)
    frame = 520
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view_begin = views[0]  # Choose a specific time for rendering
    view_end = views[-1]
    view = views[idx]

    R_begin = view_begin.R
    R_end = view_end.R
    t_begin = view_begin.T
    t_end = view_end.T

    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = view.fid

        ratio = i / (frame - 1)

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def interpolate_view_original(model_path, load2gpt_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background,
                              timer):
    render_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_hyper_view_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 1000
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    R = []
    T = []
    for view in views:
        R.append(view.R)
        T.append(view.T)

    view = views[0]
    renderings = []
    for i in tqdm(range(frame), desc="Rendering progress"):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        query_idx = i / frame * len(views)
        begin_idx = int(np.floor(query_idx))
        end_idx = int(np.ceil(query_idx))
        if end_idx == len(views):
            break
        view_begin = views[begin_idx]
        view_end = views[end_idx]
        R_begin = view_begin.R
        R_end = view_end.R
        t_begin = view_begin.T
        t_end = view_end.T

        ratio = query_idx - begin_idx

        R_cur = (1 - ratio) * R_begin + ratio * R_end
        T_cur = (1 - ratio) * t_begin + ratio * t_end

        view.reset_extrinsic(R_cur, T_cur)

        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = timer.step(xyz.detach(), time_input)

        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=60, quality=8)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        elif mode == "view":
            render_func = interpolate_view
        elif mode == "pose":
            render_func = interpolate_poses
        elif mode == "original":
            render_func = interpolate_view_original
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform)

        if not skip_test:
            render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    parser.add_argument("--txt_path", type=str)
    parser.add_argument("--txt_test_path", type=str, required=True)
    parser.add_argument("--pred_img_size", type=int, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
    opt = OmegaConf.load(args.cfg)

    dataset = model.extract(args)
    dataset.pred_img_size = args.pred_img_size
    opt.params2video.update(vars(dataset))
    test = FittingImage(opt.params2video, args.iteration)
    test_set = DataLoader(args, training=False).load_data()
    
    test.render_set('test', test_set, pipeline.extract(args))
    