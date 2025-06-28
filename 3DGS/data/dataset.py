import torch.utils.data as data
import glob
import os
import os.path as osp
import cv2
import numpy as np
import torch
from gaussian_splatting.data.correct_head_mask import correct_hair_mask
from scipy.io import loadmat
import json
import pickle as pkl
import random
import torchvision.transforms as transforms
from torchvision.transforms import v2
from PIL import Image
from transformers import CLIPImageProcessor

class DatasetValid(data.Dataset):
    def __init__(self, opt, training=True) -> None:
        super().__init__()
        
        if training:
            self.all_videos_dir = open(osp.join(opt.txt_path)).read().splitlines()
        else:
            self.all_videos_dir = open(osp.join(opt.txt_test_path)).read().splitlines()
        
        self.length_token_list = []
        self.total_length = 0
        for video_dir in self.all_videos_dir:
            img_crop_path = osp.join(video_dir, 'videos/crop')
            all_images_path = sorted([file.path for file in os.scandir(img_crop_path) if file.name.endswith('.jpg') or file.name.endswith('.png')])
            num_frames = len(all_images_path)
            print('num_frames %d' % num_frames)
            self.total_length += num_frames
            self.length_token_list.append(self.total_length)

        # self.img_ori_path = sorted(glob.glob(os.path.join(opt.img_path, '../*.jpg')))
        # self.para_3dmm_path = sorted(glob.glob(os.path.join(opt.para_3dmm_path, '*.mat')))
        self.pred_img_size = opt.pred_img_size
        self.featmap_size = 64 # opt.featmap_size
        
        #  ['background', 
        #   'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        #   'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        self.lut = np.zeros((256, ), dtype=np.uint8)
        self.lut[1:14] = 1
        # self.lut[17] = 2 # 头发
        self.lut_mouth_area = np.zeros((256, ), dtype=np.uint8)
        self.lut_mouth_area[11:14] = 1

        self.training = training
        self.opt = opt
        self.clip_length = 1 # opt.clip_length

        self.pixel_transforms = transforms.Compose([
            transforms.Resize([opt.sample_size[0], opt.sample_size[1]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.clip_image_processor = CLIPImageProcessor()

        # file_mat = loadmat(os.path.join(video_dir, 'videos/inv_render/Candy.mat'))
        # angles = file_mat['euler']
        # translation = file_mat['trans']
        # focal = file_mat['focal']
        # angles *= -1
        # ids = file_mat['id']
        # condition = np.hstack([angles, translation, focal.repeat(angles.shape[0])[..., None]], )

        # angles, translations, focals = condition[:, 0:3], condition[:, 3:6], condition[:, -1]

        # angles = angles.copy()
        # angles[:, 0] *= -1
        # angles[:, 1] *= -1

        # self.angles = torch.from_numpy(angles.copy())
        # self.translations = torch.from_numpy(translations.copy())

    def _get_data(self, index):
        """Get the seperate index location from the total index

        Args:
            index (int): index in all avaible sequeneces
        
        Returns:
            main_idx (int): index specifying which video
            sub_idx (int): index specifying what the start index in this sliced video
        """
        def fetch_data(length_list, index):
            assert index < length_list[-1]
            temp_idx = np.array(length_list) > index
            list_idx = np.where(temp_idx==True)[0][0]
            sub_idx = index
            if list_idx != 0:
                sub_idx = index - length_list[list_idx - 1]
            sub_idx -= self.clip_length // 2
            if sub_idx < 0:
                sub_idx = 0
            elif sub_idx > length_list[list_idx] - self.clip_length:
                sub_idx = length_list[list_idx] - self.clip_length
            return list_idx, sub_idx

        main_idx, sub_idx = fetch_data(self.length_token_list, index)
        return main_idx, sub_idx

    def __len__(self):
        return self.total_length

    @staticmethod
    def compute_rotation(angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1])
        zeros = torch.zeros([batch_size, 1])
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def _get_mat_vector(self, face_params_dict,
                        keys_list=['id', 'exp', 'tex', 'angle', 'gamma', 'trans']):
        """Get coefficient vector from Deep3DFace_Pytorch results

        Args:
            face_params_dict (dict): face params dictionary loaded by using loadmat function

        Returns:
            np.ndarray: (1, L)
        """

        coeff_list = []
        for key in keys_list:
            coeff_list.append(face_params_dict[key])
        
        coeff_res = np.concatenate(coeff_list, axis=1)
        return coeff_res

    def _get_face_3d_params(self, para_3dmm_path):
        '''
            id: 80, exp: 64, tex: 80, angle: 3, gamma: 27, trans: 3
            lm68: 68, 2, transform_params: 5
        '''
        face_3d_params_dict = loadmat(para_3dmm_path) # dict type
        face_3d_params = self._get_mat_vector(face_3d_params_dict, 
            keys_list=['id', 'exp', 'tex', 'angle', 'gamma', 'trans', 'transform_params'])

        return face_3d_params.astype(np.float32)

    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        image = image.astype(np.float32)
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image
    
    def __getitem__(self, index):
        main_idx, sub_start_idx = self._get_data(index)
        # sub_start_idx = 200
        choose_video = self.all_videos_dir[main_idx]
        
        iden_code = torch.from_numpy(np.load(osp.join(osp.dirname(choose_video), "id_coeff.npy"))).squeeze(0)

        sub_idx = sub_start_idx + self.clip_length // 2
        
        img_full_path = osp.join(choose_video, 'videos', f'{sub_idx:06d}.jpg')
        img = cv2.imread(img_full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # img = img.astype(np.float32) / 255.0 * 2 - 1
        # process imgs
        img_size = (self.pred_img_size, self.pred_img_size)
        gt_img_size = img.shape[0]
        if gt_img_size != self.pred_img_size:
            img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        # face_mask_path = osp.join(choose_video, 'face_mask', f'{sub_idx:06d}.png')
        # rest_mask_full = cv2.imread(face_mask_path)
        # img_full_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1)
        # img[rest_mask_full > 0] = 0.
        # rest_full_tensor = torch.from_numpy(img).permute(2, 0, 1)

        img_crop_path = osp.join(choose_video, 'videos/crop', f'{sub_idx:06d}.png')
        # img_crop_path = osp.join(choose_video, 'videos', f'{sub_idx:06d}.jpg')
        img = cv2.imread(img_crop_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.astype(np.float32) / 255.0
        # img = img.astype(np.float32) / 255.0 * 2 - 1
        # process imgs
        img_size = (self.pred_img_size, self.pred_img_size)
        gt_img_size = img.shape[0]
        if gt_img_size != self.pred_img_size:
            img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        img_crop = img.astype(np.uint8)

        face_mask_path = osp.join(choose_video, 'videos/face_mask_crop', f'{sub_idx:06d}.png')
        # face_mask_path = osp.join(choose_video, 'face_mask', f'{sub_idx:06d}.png')
        rest_mask = cv2.imread(face_mask_path)
        if rest_mask.shape[0] != self.pred_img_size:
            rest_mask = cv2.resize(rest_mask, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        mask_path = osp.join(choose_video, 'videos/parsing_crop', f'{sub_idx:06d}.png')
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        res = mask_img.astype(np.uint8)
        cv2.LUT(res, self.lut, res)
        res = correct_hair_mask(res)
        res[res != 0] = 255
        mask_img = res
        img_crop_mask = img.copy().astype(np.uint8)
        img_crop_mask[mask_img == 0.] = 0

        rest_img = img.copy()
        rest_img[rest_mask > 0] = 0.5
        # rest_img[mask_img >= 0.5] = 1.0
        # rest_img[mask_img >= 0.5] = 0.0
        # rest_tensor.append(torch.from_numpy(rest_img).permute(2, 0, 1))
        rest_tensor = torch.from_numpy(rest_img).permute(2, 0, 1)

        erode_rest_mask = rest_mask.copy()
        # # erode_rest_mask[mask_img < 0.5] = 0.0
        # k1 = np.ones((7, 7), np.uint8)
        # erode_rest_mask = cv2.erode(erode_rest_mask, k1)
        # rest_mask_tensor.append(torch.from_numpy(erode_rest_mask).permute(2, 0, 1) / 255.0)
        rest_mask_tensor = torch.from_numpy(erode_rest_mask).permute(2, 0, 1) / 255.0

        trans_params = 0
        if self.training == False:
            trans_params_path = osp.join(choose_video, 'videos/crop', f'{sub_idx:06d}.mat')
            # trans_params_path = osp.join(choose_video, 'videos/crop', f'{200:06d}.mat')
            trans_params = loadmat(trans_params_path)['trans_params'][0].astype(np.float32)
        
        img_ori_tensor = 0
        if self.training is False:
            # img_ori_path = osp.join(choose_video, 'videos', f'{sub_idx:06d}.png')
            img_ori_path = osp.join(choose_video, 'videos', f'{sub_idx:06d}.jpg')
            img_ori = cv2.imread(img_ori_path)
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            # img_ori = img_ori.astype(np.float32) / 255.0
            img_ori = img_ori.astype(np.float32) / 255.0 * 2 - 1
            img_ori_tensor = torch.from_numpy(img_ori).permute(2, 0, 1)

        base_code_tensor = []
        base_c2w_Rmat_tensor = []
        base_c2w_Tvec_tensor = []
        base_w2c_Tvec_tensor = []
        temp_inmat_tensor = []
        for idx in range(sub_start_idx, sub_start_idx+self.clip_length):
            # load init codes from the results generated by solving 3DMM rendering opt.
            para_3dmm_path = osp.join(choose_video, 'deep3dface', f'{idx:06d}.mat')
            base_code = torch.from_numpy(self._get_face_3d_params(para_3dmm_path)).squeeze(0)
            
            # para_3dmm_path_ref = osp.join(choose_video, 'deep3dface', f'{idx+700:06d}.mat')
            # base_code_ref = torch.from_numpy(self._get_face_3d_params(para_3dmm_path_ref)).squeeze(0)
            # base_code[80: 144] = base_code_ref[80: 144]
            # para_3dmm_path_0 = osp.join(choose_video, 'deep3dface', f'{200:06d}.mat')
            # base_code_0 = torch.from_numpy(self._get_face_3d_params(para_3dmm_path_0)).squeeze(0)
            # # base_code[144: 224] = base_code_0[144: 224]
            # # base_code[227: 254] = base_code_0[227: 254]
            # base_code[224: 227] = base_code_0[224: 227]
            # base_code[254: 257] = base_code_0[254: 257]

            base_w2c_Tvec = base_code[254:257]
            base_w2c_Tvec[-1] -= 10
            base_c2w_Rmat = self.compute_rotation(base_code[None, 224:227]).squeeze(0)
            base_c2w_Tvec = -(base_c2w_Rmat.matmul(base_w2c_Tvec.view(3, 1)))
            base_c2w_Tvec = base_c2w_Tvec.view(3, 1)

            # base_w2c_Tvec = self.translations[idx]
            # base_w2c_Tvec[-1] -= 10
            # base_w2c_Tvec *= 0.27
            # base_c2w_Rmat = self.compute_rotation(self.angles[idx][None]).squeeze(0)
            # base_c2w_Tvec = -(base_c2w_Rmat.matmul(base_w2c_Tvec.view(3, 1)))
            # base_c2w_Tvec = base_c2w_Tvec.view(3, 1)

            Rot = torch.eye(3)
            Rot[0, 0] = 1
            Rot[1, 1] = -1
            Rot[2, 2] = -1
            base_c2w_Rmat = base_c2w_Rmat.matmul(Rot)

            focal = 1015. * 1024 / 224 * (300 / (102*1024/224)) / 700
            center = 0.5
            # focal = 1100
            # center = 256
            inmat = torch.eye(3, dtype=torch.float32)
            inmat[0, 0] = focal
            inmat[1, 1] = focal
            inmat[0, 2] = center
            inmat[1, 2] = center

            temp_inmat = inmat
            # if self.opt.train_G and self.training:# and idx == sub_idx:
            #     scale = 3 / 32
            #     temp_inmat[:2, :2] *= 1 + random.uniform(-scale, scale)
            #     temp_inmat[0, 2] += random.uniform(-scale, scale)
            #     temp_inmat[1, 2] += random.uniform(-scale, scale)

            base_code_tensor.append(base_code)
            base_c2w_Rmat_tensor.append(base_c2w_Rmat)
            base_c2w_Tvec_tensor.append(base_c2w_Tvec)
            temp_inmat_tensor.append(temp_inmat)
            base_w2c_Tvec_tensor.append(Rot @ base_w2c_Tvec)

        base_code = torch.stack(base_code_tensor, dim=0)
        base_c2w_Rmat = torch.stack(base_c2w_Rmat_tensor, dim=0)
        base_c2w_Tvec = torch.stack(base_c2w_Tvec_tensor, dim=0)
        temp_inmat = torch.stack(temp_inmat_tensor, dim=0)
        base_w2c_Tvec = torch.stack(base_w2c_Tvec_tensor, dim=0)

        # lm68_path = osp.join(choose_video, 'videos/crop', f'{sub_idx:06d}.txt')
        # lm68 = np.loadtxt(lm68_path).reshape(68, -1)
        lm68 = 0

        ref_img_idx = random.randint(0, self.total_length - 1)
        ref_img = cv2.imread(osp.join(choose_video, 'videos/crop', f'{ref_img_idx:06d}.png'))
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = self.contrast_normalization(ref_img)
        if ref_img.shape[0] != self.pred_img_size:
            ref_img = cv2.resize(ref_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        
        mask_path = osp.join(choose_video, 'videos/parsing_crop', f'{ref_img_idx:06d}.png')
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        res = mask_img.astype(np.uint8)
        cv2.LUT(res, self.lut, res)
        res = correct_hair_mask(res)
        res[res != 0] = 255
        mask_img = res
    
        pixel_values_ref_pose = ref_img.copy().astype(np.uint8)
        pixel_values_ref_pose[mask_img == 0.] = 0.

        batch_dict = {
            'idx': index,
            # 'img_full': img_full_tensor,
            'img_crop': img_crop,
            'rest_mask': rest_mask_tensor,
            'base_code': base_code,
            'cam_info': {
                "batch_c2w_Rmats": base_c2w_Rmat,
                "batch_w2c_Tvecs": base_w2c_Tvec,
                "batch_inmats": temp_inmat,
                'FoVx': 2 * np.arctan(center / focal),
                'FoVy': 2 * np.arctan(center / focal),
            },
            'trans_params': trans_params,
            'img_ori': img_ori_tensor,
            'iden': iden_code,
            'rest_img': rest_tensor,
            # 'rest_img_full': rest_full_tensor,
            'lm68': lm68,
            'img_crop_mask': img_crop_mask,
            'ref_img': ref_img,
            'pixel_values_ref_pose': pixel_values_ref_pose,
        }

        return batch_dict
    

class Dataset(data.Dataset):
    def __init__(self, opt, training=True) -> None:
        super().__init__()
        
        if training:
            self.all_videos_dir = open(osp.join(opt.txt_path)).read().splitlines()
        else:
            self.all_videos_dir = open(osp.join(opt.txt_test_path)).read().splitlines()
        
        self.length_token_list = []
        self.total_length = 0
        for video_dir in self.all_videos_dir:
            img_crop_path = osp.join(video_dir, 'videos/crop')
            all_images_path = sorted([file.path for file in os.scandir(img_crop_path) if file.name.endswith('.jpg') or file.name.endswith('.png')])
            num_frames = len(all_images_path)
            print('num_frames %d' % num_frames)
            self.total_length += num_frames
            self.length_token_list.append(self.total_length)

        # self.img_ori_path = sorted(glob.glob(os.path.join(opt.img_path, '../*.jpg')))
        # self.para_3dmm_path = sorted(glob.glob(os.path.join(opt.para_3dmm_path, '*.mat')))
        self.pred_img_size = opt.pred_img_size
        self.featmap_size = 64 # opt.featmap_size
        
        #  ['background', 
        #   'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        #   'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        self.lut = np.zeros((256, ), dtype=np.uint8)
        self.lut[1:14] = 1
        # self.lut[17] = 2 # 头发
        self.lut_mouth_area = np.zeros((256, ), dtype=np.uint8)
        self.lut_mouth_area[11:14] = 1

        self.training = training
        self.opt = opt
        self.clip_length = opt.clip_length

        self.pixel_transforms = transforms.Compose([
            transforms.Resize([opt.sample_size[0], opt.sample_size[1]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.clip_image_processor = CLIPImageProcessor()

        # file_mat = loadmat(os.path.join(video_dir, 'videos/inv_render/Candy.mat'))
        # angles = file_mat['euler']
        # translation = file_mat['trans']
        # focal = file_mat['focal']
        # angles *= -1
        # ids = file_mat['id']
        # condition = np.hstack([angles, translation, focal.repeat(angles.shape[0])[..., None]], )

        # angles, translations, focals = condition[:, 0:3], condition[:, 3:6], condition[:, -1]

        # angles = angles.copy()
        # angles[:, 0] *= -1
        # angles[:, 1] *= -1

        # self.angles = torch.from_numpy(angles.copy())
        # self.translations = torch.from_numpy(translations.copy())

    def _get_data(self, index):
        """Get the seperate index location from the total index

        Args:
            index (int): index in all avaible sequeneces
        
        Returns:
            main_idx (int): index specifying which video
            sub_idx (int): index specifying what the start index in this sliced video
        """
        def fetch_data(length_list, index):
            assert index < length_list[-1]
            temp_idx = np.array(length_list) > index
            list_idx = np.where(temp_idx==True)[0][0]
            sub_idx = index
            if list_idx != 0:
                sub_idx = index - length_list[list_idx - 1]
            sub_idx -= self.clip_length // 2
            if sub_idx < 0:
                sub_idx = 0
            elif sub_idx > length_list[list_idx] - self.clip_length:
                sub_idx = length_list[list_idx] - self.clip_length
            return list_idx, sub_idx

        main_idx, sub_idx = fetch_data(self.length_token_list, index)
        return main_idx, sub_idx

    def __len__(self):
        return self.total_length

    @staticmethod
    def compute_rotation(angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1])
        zeros = torch.zeros([batch_size, 1])
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def _get_mat_vector(self, face_params_dict,
                        keys_list=['id', 'exp', 'tex', 'angle', 'gamma', 'trans']):
        """Get coefficient vector from Deep3DFace_Pytorch results

        Args:
            face_params_dict (dict): face params dictionary loaded by using loadmat function

        Returns:
            np.ndarray: (1, L)
        """

        coeff_list = []
        for key in keys_list:
            coeff_list.append(face_params_dict[key])
        
        coeff_res = np.concatenate(coeff_list, axis=1)
        return coeff_res

    def _get_face_3d_params(self, para_3dmm_path):
        '''
            id: 80, exp: 64, tex: 80, angle: 3, gamma: 27, trans: 3
            lm68: 68, 2, transform_params: 5
        '''
        face_3d_params_dict = loadmat(para_3dmm_path) # dict type
        face_3d_params = self._get_mat_vector(face_3d_params_dict, 
            keys_list=['id', 'exp', 'tex', 'angle', 'gamma', 'trans', 'transform_params'])

        return face_3d_params.astype(np.float32)

    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        image = image.astype(np.float32)
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image
    
    def __getitem__(self, index):
        main_idx, sub_start_idx = self._get_data(index)
        # sub_start_idx = 200
        choose_video = self.all_videos_dir[main_idx]
        
        iden_code = torch.from_numpy(np.load(osp.join(osp.dirname(choose_video), "id_coeff.npy"))).squeeze(0)
        
        img_tensor_list = []
        mask_img_tensor_list = []
        mask_tensor_list = []
        for sub_idx in range(sub_start_idx, sub_start_idx+self.clip_length):
            img_full_path = osp.join(choose_video, 'videos', f'{sub_idx:06d}.jpg')
            img = cv2.imread(img_full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            # img = img.astype(np.float32) / 255.0 * 2 - 1
            # process imgs
            img_size = (self.pred_img_size, self.pred_img_size)
            gt_img_size = img.shape[0]
            if gt_img_size != self.pred_img_size:
                img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            # face_mask_path = osp.join(choose_video, 'face_mask', f'{sub_idx:06d}.png')
            # rest_mask_full = cv2.imread(face_mask_path)
            img_full_tensor = torch.from_numpy(img).permute(2, 0, 1)
            # img[rest_mask_full > 0] = 0.
            # rest_full_tensor = torch.from_numpy(img).permute(2, 0, 1)
            # img_tensor_list.append(img_full_tensor)

            mask_path = osp.join(choose_video, 'videos/parsing', f'{sub_idx:06d}.png')
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            if mask_img.shape[0] != self.pred_img_size:
                mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            res = mask_img.astype(np.uint8)
            res_mouth_area = res.copy()
            cv2.LUT(res, self.lut, res)
            res = correct_hair_mask(res)
            res[res != 0] = 255
            mask_img = res
            # mask_tensor.append(torch.from_numpy(mask_img[None, :, :]))
            mask_tensor = torch.from_numpy(mask_img[None, :, :])
            mask_tensor_list.append(mask_tensor)
            
            # RR = transforms.RandomRotation(degrees=(-90, 90))
            # img_crop_mask_tensor = RR(img_crop_mask_tensor)
            # img_crop_mask_tensor = v2.RandomAffine(degrees=0, translate=(0.0, 0.1), scale=(0.8, 1.0))(img_crop_mask_tensor)
            cv2.LUT(res_mouth_area, self.lut_mouth_area, res_mouth_area)
            res_mouth_area = correct_hair_mask(res_mouth_area)
            res_mouth_area[res_mouth_area != 0] = 1
            mask_img_mouth_area = res_mouth_area
            # mask_tensor.append(torch.from_numpy(mask_img[None, :, :]))
            mask_mouth_area_tensor = torch.from_numpy(mask_img_mouth_area[None, :, :])
            mask_img_full_tensor = img_full_tensor.clone()
            mask_img_full_tensor[mask_tensor.repeat(3, 1, 1) == 0.] = 0. # -1.
            # mask_img_tensor_list.append(mask_img_full_tensor)

            img_crop_path = osp.join(choose_video, 'videos/crop', f'{sub_idx:06d}.png')
            img = cv2.imread(img_crop_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            # img = img.astype(np.float32) / 255.0 * 2 - 1
            # process imgs
            img_size = (self.pred_img_size, self.pred_img_size)
            gt_img_size = img.shape[0]
            if gt_img_size != self.pred_img_size:
                img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
            img_crop_tensor = torch.from_numpy(img).permute(2, 0, 1)
            img_tensor_list.append(img_crop_tensor)

            mask_path = osp.join(choose_video, 'videos/parsing_crop', f'{sub_idx:06d}.png')
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
            if mask_img.shape[0] != self.pred_img_size:
                mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            res = mask_img.astype(np.uint8)
            res_mouth_area = res.copy()
            cv2.LUT(res, self.lut, res)
            res = correct_hair_mask(res)
            res[res != 0] = 255
            mask_img = res
            # mask_tensor.append(torch.from_numpy(mask_img[None, :, :]))
            mask_tensor = torch.from_numpy(mask_img[None, :, :])
            
            # RR = transforms.RandomRotation(degrees=(-90, 90))
            # img_crop_mask_tensor = RR(img_crop_mask_tensor)
            # img_crop_mask_tensor = v2.RandomAffine(degrees=0, translate=(0.0, 0.1), scale=(0.8, 1.0))(img_crop_mask_tensor)
            cv2.LUT(res_mouth_area, self.lut_mouth_area, res_mouth_area)
            res_mouth_area = correct_hair_mask(res_mouth_area)
            res_mouth_area[res_mouth_area != 0] = 1
            mask_img_mouth_area = res_mouth_area
            # mask_tensor.append(torch.from_numpy(mask_img[None, :, :]))
            mask_mouth_area_tensor = torch.from_numpy(mask_img_mouth_area[None, :, :])
            mask_img_crop_tensor = img_crop_tensor.clone()
            mask_img_crop_tensor[mask_tensor.repeat(3, 1, 1) == 0.] = 0. # -1.
            # mask_img_tensor_list.append(mask_img_crop_tensor)

            face_mask_path = osp.join(choose_video, 'videos/face_mask_crop', f'{sub_idx:06d}.png')
            # face_mask_path = osp.join(choose_video, 'face_mask', f'{sub_idx:06d}.png')
            rest_mask = cv2.imread(face_mask_path)
            if rest_mask.shape[0] != self.pred_img_size:
                rest_mask = cv2.resize(rest_mask, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

            rest_img = img.copy()
            rest_img[rest_mask > 0] = 0.5
            # rest_img[mask_img >= 0.5] = 1.0
            # rest_img[mask_img >= 0.5] = 0.0
            # rest_tensor.append(torch.from_numpy(rest_img).permute(2, 0, 1))
            rest_tensor = torch.from_numpy(rest_img).permute(2, 0, 1)
            mask_img_tensor_list.append(rest_tensor)

            erode_rest_mask = rest_mask.copy()
            # # erode_rest_mask[mask_img < 0.5] = 0.0
            # k1 = np.ones((7, 7), np.uint8)
            # erode_rest_mask = cv2.erode(erode_rest_mask, k1)
            # rest_mask_tensor.append(torch.from_numpy(erode_rest_mask).permute(2, 0, 1) / 255.0)
            rest_mask_tensor = torch.from_numpy(erode_rest_mask).permute(2, 0, 1) / 255.0

            trans_params = 0
            if self.training == False:
                trans_params_path = osp.join(choose_video, 'videos/crop', f'{sub_idx:06d}.mat')
                # trans_params_path = osp.join(choose_video, 'videos/crop', f'{200:06d}.mat')
                trans_params = loadmat(trans_params_path)['trans_params'][0].astype(np.float32)
            
            img_ori_tensor = 0
            if self.training is False:
                # img_ori_path = osp.join(choose_video, 'videos', f'{sub_idx:06d}.png')
                img_ori_path = osp.join(choose_video, 'videos', f'{sub_idx:06d}.jpg')
                img_ori = cv2.imread(img_ori_path)
                img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
                # img_ori = img_ori.astype(np.float32) / 255.0
                img_ori = img_ori.astype(np.float32) / 255.0 * 2 - 1
                img_ori_tensor = torch.from_numpy(img_ori).permute(2, 0, 1)

        base_code_tensor = []
        base_c2w_Rmat_tensor = []
        base_c2w_Tvec_tensor = []
        base_w2c_Tvec_tensor = []
        temp_inmat_tensor = []
        for idx in range(sub_start_idx, sub_start_idx+self.clip_length):
            # load init codes from the results generated by solving 3DMM rendering opt.
            para_3dmm_path = osp.join(choose_video, 'deep3dface', f'{idx:06d}.mat')
            base_code = torch.from_numpy(self._get_face_3d_params(para_3dmm_path)).squeeze(0)
            
            # para_3dmm_path_ref = osp.join(choose_video, 'deep3dface', f'{idx+700:06d}.mat')
            # base_code_ref = torch.from_numpy(self._get_face_3d_params(para_3dmm_path_ref)).squeeze(0)
            # base_code[80: 144] = base_code_ref[80: 144]
            # para_3dmm_path_0 = osp.join(choose_video, 'deep3dface', f'{200:06d}.mat')
            # base_code_0 = torch.from_numpy(self._get_face_3d_params(para_3dmm_path_0)).squeeze(0)
            # # base_code[144: 224] = base_code_0[144: 224]
            # # base_code[227: 254] = base_code_0[227: 254]
            # base_code[224: 227] = base_code_0[224: 227]
            # base_code[254: 257] = base_code_0[254: 257]

            base_w2c_Tvec = base_code[254:257]
            base_w2c_Tvec[-1] -= 10
            base_c2w_Rmat = self.compute_rotation(base_code[None, 224:227]).squeeze(0)
            base_c2w_Tvec = -(base_c2w_Rmat.matmul(base_w2c_Tvec.view(3, 1)))
            base_c2w_Tvec = base_c2w_Tvec.view(3, 1)

            # base_w2c_Tvec = self.translations[idx]
            # base_w2c_Tvec[-1] -= 10
            # # base_w2c_Tvec *= 0.27
            # # base_w2c_Tvec[1] += 0.015
            # # base_w2c_Tvec[2] += 0.161
            # # base_w2c_Tvec = base_w2c_Tvec / 4.0 * 2.7
            # base_c2w_Rmat = self.compute_rotation(self.angles[idx][None]).squeeze(0)
            # base_c2w_Tvec = -(base_c2w_Rmat.matmul(base_w2c_Tvec.view(3, 1)))
            # base_c2w_Tvec = base_c2w_Tvec.view(3, 1)

            Rot = torch.eye(3)
            Rot[0, 0] = 1
            Rot[1, 1] = -1
            Rot[2, 2] = -1
            base_c2w_Rmat = base_c2w_Rmat.matmul(Rot)

            focal = 1015. * 1024 / 224 * (300 / (102*1024/224)) / 700
            center = 0.5
            # focal = 2985.29 / 700 * 1100 / 1050
            # center = 0.5
            inmat = torch.eye(3, dtype=torch.float32)
            inmat[0, 0] = focal
            inmat[1, 1] = focal
            inmat[0, 2] = center
            inmat[1, 2] = center

            temp_inmat = inmat
            # if self.opt.train_G and self.training:# and idx == sub_idx:
            #     scale = 3 / 32
            #     temp_inmat[:2, :2] *= 1 + random.uniform(-scale, scale)
            #     temp_inmat[0, 2] += random.uniform(-scale, scale)
            #     temp_inmat[1, 2] += random.uniform(-scale, scale)

            base_code_tensor.append(base_code)
            base_c2w_Rmat_tensor.append(base_c2w_Rmat)
            base_c2w_Tvec_tensor.append(base_c2w_Tvec)
            temp_inmat_tensor.append(temp_inmat)
            base_w2c_Tvec_tensor.append(Rot @ base_w2c_Tvec)

        base_code = torch.stack(base_code_tensor, dim=0)
        base_c2w_Rmat = torch.stack(base_c2w_Rmat_tensor, dim=0)
        base_c2w_Tvec = torch.stack(base_c2w_Tvec_tensor, dim=0)
        temp_inmat = torch.stack(temp_inmat_tensor, dim=0)
        base_w2c_Tvec = torch.stack(base_w2c_Tvec_tensor, dim=0)

        # lm68_path = osp.join(choose_video, 'videos/crop', f'{sub_idx:06d}.txt')
        # lm68 = np.loadtxt(lm68_path).reshape(68, -1)
        lm68 = 0

        ref_img_idx = random.randint(0, self.total_length - 1)
        ref_img = cv2.imread(osp.join(choose_video, 'videos/crop', f'{ref_img_idx:06d}.png'))
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = self.contrast_normalization(ref_img)
        ref_img_pil = Image.fromarray(ref_img)
        
        clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values

        pixel_values_ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).contiguous()
        pixel_values_ref_img = pixel_values_ref_img / 255.
        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
        pixel_values_ref_img = self.pixel_transforms(pixel_values_ref_img) # -1 ~ 1
        pixel_values_ref_img = pixel_values_ref_img.squeeze(0)
        
        mask_path = osp.join(choose_video, 'videos/parsing_crop', f'{ref_img_idx:06d}.png')
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        res = mask_img.astype(np.uint8)
        cv2.LUT(res, self.lut, res)
        res = correct_hair_mask(res)
        res[res != 0] = 255
        mask_img = res
        mask_tensor = torch.from_numpy(mask_img[None, :, :])
    
        pixel_values_ref_pose = pixel_values_ref_img.clone()
        pixel_values_ref_pose[mask_tensor.repeat(3, 1, 1) == 0.] = -1.

        drop_image_embeds = 1 if random.random() < 0.1 else 0

        batch_dict = {
            # 'idx': index,
            # 'img_full': img_full_tensor,
            'img': torch.stack(img_tensor_list, dim=0),
            'mask_img': torch.stack(mask_img_tensor_list, dim=0),
            'mask': torch.stack(mask_tensor_list, dim=0),
            # 'rest_mask': rest_mask_tensor,
            'base_code': base_code,
            'cam_info': {
                "batch_c2w_Rmats": base_c2w_Rmat,
                "batch_w2c_Tvecs": base_w2c_Tvec,
                "batch_inmats": temp_inmat,
                'FoVx': 2 * np.arctan(center / focal),
                'FoVy': 2 * np.arctan(center / focal),
            },
            # 'trans_params': trans_params,
            # 'img_ori': img_ori_tensor,
            # 'iden': iden_code,
            # 'rest_img': rest_tensor,
            # # 'rest_img_full': rest_full_tensor,
            # 'lm68': lm68,
            # 'mouth_area_mask': mask_mouth_area_tensor,
            # 'clip_ref_image': clip_ref_image,
            # 'pixel_values_ref_img': pixel_values_ref_img,
            # 'pixel_values_ref_pose': pixel_values_ref_pose,
            # 'drop_image_embeds': drop_image_embeds,
        }

        return batch_dict
