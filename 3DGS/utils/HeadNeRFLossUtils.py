import cv2
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import os.path as osp
from scipy.io import loadmat
from NetWorks import networks_modified as networks
from Deep3DFaceRecon_pytorch.models import create_model
from Deep3DFaceRecon_pytorch.models.facerecon_model import FaceReconModel
from Deep3DFaceRecon_pytorch.models.networks import define_net_recon
from kornia.geometry import warp_affine
from skimage import transform as trans
# from Utils.load_model import load_state_dict

# import lpips
# class VGGPerceptualLoss(torch.nn.Module):
#     def __init__(self, resize=True):
#         super(VGGPerceptualLoss, self).__init__()
#         self.vgg_loss_fn = lpips.LPIPS(net='vgg')


#     def forward(self, input, target,):
#         res = self.vgg_loss_fn(input, target)
#         return res.mean()

# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_folder):
    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm'] # (68, 3)

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1 # (7)
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0) # (5, 3)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :] # (5, 3)

    return Lm3D

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    

class HeadNeRFLossUtils(object):
    def __init__(self, opt, use_vgg_loss=True, device=None) -> None:
        super().__init__()
        
        if opt.white_background:
            self.bg_value = 1.0
        else:
            self.bg_value = 0.0
            
        self.use_vgg_loss = use_vgg_loss
        if self.use_vgg_loss:
            assert device is not None
            self.device = device
            self.vgg_loss_func = VGGPerceptualLoss(resize=True).to(self.device)

        no_lsgan = False
        self.criterionGAN = networks.GANLoss(use_lsgan=not no_lsgan, tensor=torch.cuda.FloatTensor)
        self.criterionFeat = torch.nn.L1Loss()
        self.opt = opt

        # model = create_model(opt)
        # self.net_recon = define_net_recon(
        #     net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path
        # )
        # state_dict = torch.load('Deep3DFaceRecon_pytorch/checkpoints/face_recon_feat0.2_augment/epoch_20.pth', map_location=torch.device("cpu"))
        # load_state_dict(self.net_recon, state_dict['net_recon'])
        # self.net_recon.eval()
        # self.net_recon.to(self.device)
        # self.lm3d_std = load_lm3d("Deep3DFaceRecon_pytorch/BFM") # (5, 3)

    @staticmethod
    def calc_cam_loss(delta_cam_info):
        delta_eulur_loss = torch.mean(delta_cam_info["delta_eulur"] * delta_cam_info["delta_eulur"])
        delta_tvec_loss = torch.mean(delta_cam_info["delta_tvec"] * delta_cam_info["delta_tvec"])
        
        return {
            "delta_eular": delta_eulur_loss, 
            "delta_tvec": delta_tvec_loss
        }
        
        
    def calc_code_loss(self, opt_code_dict):
        # iden_code_opt = opt_code_dict["iden"]
        expr_code_opt = opt_code_dict["expr"]

        # iden_loss = torch.mean(iden_code_opt * iden_code_opt)
        expr_loss = torch.mean(expr_code_opt * expr_code_opt)
        
        appea_loss = torch.mean(opt_code_dict["appea"] * opt_code_dict["appea"])

        bg_code = opt_code_dict["bg"]
        if bg_code is None:
            bg_loss = torch.as_tensor(0.0, dtype=expr_loss.dtype, device=expr_loss.device)
        else:
            bg_loss = torch.mean(bg_code * bg_code)
            
        res_dict = {
            # "iden_code": iden_loss,
            "expr_code": expr_loss,
            "appea_code": appea_loss,
            "bg_code": bg_loss
        }
        
        return res_dict
    
    
    def calc_data_loss(self, data_dict, gt_rgb, head_mask_c1b, nonhead_mask_c1b):
        bg_value = self.bg_value
        
        # bg_img = data_dict["bg_img"]
        # bg_loss = torch.mean((bg_img - bg_value) * (bg_img - bg_value))
        
        res_img = data_dict["image"]
        # res_img_inter = data_dict["image_inter"]
        if self.opt.train_G:
            head_loss = F.mse_loss(res_img, gt_rgb)
            nonhaed_loss = 0
        else:
            head_mask_c3b = head_mask_c1b.expand(-1, 3, -1, -1)
            head_loss = F.mse_loss(res_img[head_mask_c3b], gt_rgb[head_mask_c3b]) #+ F.mse_loss(res_img_inter[head_mask_c3b], gt_rgb[head_mask_c3b])

            nonhead_mask_c3b = nonhead_mask_c1b.expand(-1, 3, -1, -1)
            temp_tensor = res_img[nonhead_mask_c3b]
            tv = temp_tensor - bg_value
            # temp_tensor_inter = res_img_inter[nonhead_mask_c3b]
            # tv_inter = temp_tensor_inter - bg_value
            nonhaed_loss = torch.mean(tv * tv) #+ torch.mean(tv_inter * tv_inter)
        # head_loss = F.mse_loss(res_img, gt_rgb)
        # nonhaed_loss = 0
        res = {
            "bg_loss": 0, #bg_loss,
            "head_loss": head_loss,  
            "nonhaed_loss": nonhaed_loss, 
        }

        if self.use_vgg_loss:
            masked_gt_img = gt_rgb.clone()
            if not self.opt.train_G:
                masked_gt_img[~head_mask_c3b] = bg_value
            
            temp_res_img = res_img
            vgg_loss = self.vgg_loss_func(temp_res_img, masked_gt_img) #+ self.vgg_loss_func(res_img_inter, masked_gt_img)
            res["vgg"] = vgg_loss

        return res
    
    # calculating least square problem for image alignment
    def POS(self, xp, x):
        npts = xp.shape[1] # 5

        A = np.zeros([2*npts, 8]) # (10, 8)

        A[0:2*npts-1:2, 0:3] = x.transpose()
        A[0:2*npts-1:2, 3] = 1

        A[1:2*npts:2, 4:7] = x.transpose()
        A[1:2*npts:2, 7] = 1

        b = np.reshape(xp.transpose(), [2*npts, 1]) # (10, 1)

        k, _, _, _ = np.linalg.lstsq(A, b)

        R1 = k[0:3] # (3, 1)
        R2 = k[4:7] # (3, 1)
        sTx = k[3] # (1)
        sTy = k[7] # (1)
        s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
        t = np.stack([sTx, sTy], axis=0) # (2, 1)

        return t, s

    # # utils for face reconstruction
    # def extract_5p(self, lm):
    #     lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    #     lm5p = torch.stack([lm[:, lm_idx[0], :], torch.mean(lm[:, lm_idx[[1, 2]], :], 1), torch.mean(
    #         lm[:, lm_idx[[3, 4]], :], 1), lm[:, lm_idx[5], :], lm[:, lm_idx[6], :]], dim=1)
    #     lm5p = lm5p[:, [1, 2, 0, 3, 4], :]
    #     return lm5p

    # utils for face reconstruction
    def extract_5p(self, lm):
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
            lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
        lm5p = lm5p[[1, 2, 0, 3, 4], :]
        return lm5p

    # # resize and crop images for face reconstruction
    # def resize_n_crop_img(self, img, lm, t, s, target_size=224., 
    #         mask=None, normal=None, parsing=None):
    #     w0, h0 = img.size
    #     w = (w0*s).astype(np.int32)
    #     h = (h0*s).astype(np.int32)
    #     left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    #     right = left + target_size
    #     up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    #     below = up + target_size

    #     img = img.resize((w, h), resample=Image.BICUBIC)
    #     img = img.crop((left, up, right, below))

    #     if mask is not None:
    #         mask = mask.resize((w, h), resample=Image.BICUBIC)
    #         mask = mask.crop((left, up, right, below))

    #     if normal is not None:
    #         normal = normal.resize((w, h), resample=Image.BICUBIC)
    #         normal = normal.crop((left, up, right, below))

    #     if parsing is not None:
    #         parsing = parsing.resize((w, h), resample=Image.NEAREST)
    #         parsing = parsing.crop((left, up, right, below))

    #     lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
    #                 t[1] + h0/2], axis=1)*s # (5, 2)
    #     lm = lm - np.reshape(
    #             np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2]) # (5, 2)

    #     return img, lm, mask, normal, parsing

    # resize and crop images for face reconstruction
    def resize_n_crop_img(self, img, t, s, target_size=224.):
        B = len(img)
        w0, h0 = img.shape[2:]
        w = (w0*s).to(torch.int32)
        h = (h0*s).to(torch.int32)
        left = (w/2 - target_size/2 + ((t[:, 0] - w0/2)*s)).to(torch.int32)
        right = (left + target_size).to(torch.int32)
        up = (h/2 - target_size/2 + ((h0/2 - t[:, 1])*s)).to(torch.int32)
        below = (up + target_size).to(torch.int32)

        img_list = []
        for i in range(B):
            img_ = F.interpolate(img[i:i+1], (w[i], h[i]), mode='bicubic')
            img_0 = torch.zeros((1, 3, int(target_size), int(target_size))).to(img.device)

            if up[i] >= 0:
                up_ = up[i]
                up_0 = 0
            else:
                up_ = 0
                up_0 = -up[i]
            if below[i] <= h[i]:
                below_ = below[i]
                below_0 = target_size
            else:
                below_ = h[i]
                below_0 = target_size - (below[i] - h[i])
            
            if left[i] >= 0:
                left_ = left[i]
                left_0 = 0
            else:
                left_ = 0
                left_0 = -left[i]
            if right[i] <= w[i]:
                right_ = right[i]
                right_0 = target_size
            else:
                right_ = w[i]
                right_0 = target_size - (right[i] - w[i])

            img_0[:, :, int(up_0):int(below_0), int(left_0):int(right_0)] = \
                img_[:, :, int(up_):int(below_), int(left_):int(right_)]
            # img_ = img_[:, :, up[i]:below[i], left[i]:right[i]]
            img_list.append(img_0)
        img = torch.cat(img_list, dim=0)

        return img

    def resize_n_crop(self, image, M, dsize=112):
        # image: (b, c, h, w)
        # M   :  (b, 2, 3)
        return warp_affine(image, M, dsize=(dsize, dsize))

    # utils for face recognition model
    def estimate_norm(self, lm_68p, H, lm3D):
        # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
        """
        Return:
            trans_m            --numpy.array  (2, 3)
        Parameters:
            lm                 --numpy.array  (68, 2), y direction is opposite to v direction
            H                  --int/float , image height
        """
        lm = self.extract_5p(lm_68p)
        lm[:, -1] = H - 1 - lm[:, -1]
        tform = trans.SimilarityTransform()
        # src = np.array(
        # [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        # [41.5493, 92.3655], [70.7299, 92.2041]],
        # dtype=np.float32)
        src = lm3D[:, :2]
        tform.estimate(lm, src)
        M = tform.params
        if np.linalg.det(M) == 0:
            M = np.eye(3)

        return M[0:2, :]

    def estimate_norm_torch(self, lm_68p, H, lm3D):
        lm_68p_ = lm_68p.detach().cpu().numpy()
        M = []
        for i in range(lm_68p_.shape[0]):
            M.append(self.estimate_norm(lm_68p_[i], H, lm3D))
        M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_68p.device)
        return M

    # utils for face reconstruction
    def align_img(self, img, lm, lm3D, target_size=224., rescale_factor=102.):
        """
        Return:
            transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
            img_new            --PIL.Image  (target_size, target_size, 3)
            lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
            mask_new           --PIL.Image  (target_size, target_size)
        
        Parameters:
            img                --PIL.Image  (raw_H, raw_W, 3)
            lm                 --numpy.array  (68, 2), y direction is opposite to v direction
            lm3D               --numpy.array  (5, 3)
            mask               --PIL.Image  (raw_H, raw_W, 3)
        """
        lm_68p_ = lm.detach().cpu().numpy()
        t_list, s_list = [], []
        for i in range(lm_68p_.shape[0]):
            if lm_68p_[i].shape[1] != 5:
                lm5p = self.extract_5p(lm_68p_[i])
            else:
                lm5p = lm_68p_[i]
            
            # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
            t, s = self.POS(lm5p.transpose(), lm3D.transpose())
            s = rescale_factor / s

            t_list.append(t)
            s_list.append(s)

        t = torch.tensor(np.array(t_list)).squeeze().to(lm.device)
        s = torch.tensor(np.array(s_list)).to(lm.device)
        # if lm.shape[1] != 5:
        #     lm5p = self.extract_5p(lm)
        # else:
        #     lm5p = lm

        # # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
        # t, s = self.POS(lm5p.transpose(), lm3D.transpose())
        # s = rescale_factor / s

        # processing the image
        # img_new, lm_new, mask_new, normal_new, parsing_new = self.resize_n_crop_img(img, lm, t, s, target_size=target_size, 
        #     mask=mask, normal=normal, parsing=parsing)
        img_new = self.resize_n_crop_img(img, t, s, target_size=target_size)

        # trans_m_pred = self.estimate_norm_torch(lm, img.shape[-2], lm3D)
        # image = self.resize_n_crop(img, trans_m_pred, 224)
        # cv2.imwrite("0.png", img_new.detach().cpu()[0].permute(1, 2, 0).numpy()[:, :, ::-1]*255)
        pre_params, _ = self.net_recon(img_new)

        return pre_params

    def calc_total_loss(self, delta_cam_info=None, opt_code_dict=None, pred_dict=None, gt_rgb=None, mask_tensor=None, netD=None):
        # assert delta_cam_info is not None
        head_mask = (mask_tensor >= 0.5)
        nonhead_mask = (mask_tensor < 0.5)
        head_mask_c3b = head_mask.expand(-1, 3, -1, -1)

        loss_dict = {
            'head_loss': torch.tensor(0).to(self.device),
            'expr': torch.tensor(0).to(self.device),
            'appea': torch.tensor(0).to(self.device),
        }
        
        loss_dict.update(self.calc_data_loss(pred_dict, gt_rgb, head_mask, nonhead_mask))

        total_loss = 0.0
        for k in loss_dict:
            total_loss += loss_dict[k]

        # cam loss
        if delta_cam_info is not None:
            loss_dict.update(self.calc_cam_loss(delta_cam_info))
            total_loss += 0.001 * loss_dict["delta_eular"] + 0.001 * loss_dict["delta_tvec"]

        # code loss
        # loss_dict.update(self.calc_code_loss(opt_code_dict))
        # total_loss += 0.001 * loss_dict["iden_code"] + \
        #               1.0 * loss_dict["expr_code"] + \
        #               0.001 * loss_dict["appea_code"] + \
        #               0.01 * loss_dict["bg_code"]
        
        # total_loss += 1.0 * loss_dict["expr_code"] + \
        #               0.001 * loss_dict["appea_code"] + \
        #               0.01 * loss_dict["bg_code"]
        
        loss_dict["total_loss"] = total_loss

        if netD is not None:
            # Fake Detection and Loss
            y1 = pred_dict["image"].clone()
            # if not self.opt.train_G:
            #     y1[~head_mask_c3b] = self.bg_value
            pred_fake_pool = netD(y1.detach())
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss
            gt1 = gt_rgb.clone()
            if not self.opt.train_G:
                gt1[~head_mask_c3b] = self.bg_value
            pred_real = netD(gt1.detach())
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Possibility Loss)
            pred_fake = netD(y1)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            
            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            # if not self.opt.no_ganFeat_loss:
            n_layers_D = 3
            num_D = 3
            lambda_feat = 10.0
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * lambda_feat * 0.2

            loss_dict["total_loss"] += loss_G_GAN + loss_G_GAN_Feat
            loss_dict.update({"loss_D": loss_D_fake + loss_D_real,
                "D_fake": loss_D_fake,
                "D_real": loss_D_real,})

            # # Fake Detection and Loss
            # y1 = pred_dict["image"].clone()
            # # if not self.opt.train_G:
            # #     y1[~head_mask_c3b] = self.bg_value
            # pred_fake_pool = netD(y1.detach())
            # loss_D_fake = torch.nn.functional.softplus(pred_fake_pool).mean()

            # # Real Detection and Loss
            # gt1 = gt_rgb.clone()
            # if not self.opt.train_G:
            #     gt1[~head_mask_c3b] = self.bg_value
            # pred_real = netD(gt1.detach())
            # loss_D_real = torch.nn.functional.softplus(-pred_real).mean()

            # # GAN loss (Fake Possibility Loss)
            # pred_fake = netD(y1)
            # loss_G_GAN = torch.nn.functional.softplus(-pred_fake).mean()

            # loss_dict["total_loss"] += loss_G_GAN
            # loss_dict.update({"loss_D": loss_D_fake + loss_D_real,
            #     "D_fake": loss_D_fake,
            #     "D_real": loss_D_real,})
            
        return loss_dict
        