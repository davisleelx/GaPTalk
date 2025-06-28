import os
import json
from scipy.io import loadmat
import numpy as np
import torch

def get_mat_vector(face_params_dict,
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

def get_face_3d_params(para_3dmm_path):
        '''
            id: 80, exp: 64, tex: 80, angle: 3, gamma: 27, trans: 3
            lm68: 68, 2, transform_params: 5
        '''
        face_3d_params_dict = loadmat(para_3dmm_path) # dict type
        face_3d_params = get_mat_vector(face_3d_params_dict, 
            keys_list=['id', 'exp', 'tex', 'angle', 'gamma', 'trans', 'transform_params'])

        return face_3d_params.astype(np.float32)

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

def process_camera_inv(translation, Rs, focals): #crop_params):

    c_list = []

    N = len(translation)
    # for trans, R, crop_param in zip(translation,Rs, crop_params):
    for idx, (trans, R, focal) in enumerate(zip(translation, Rs, focals)):

        # idx_prev = max(idx - 1, 0)
        # idx_last = min(idx + 2, N - 1)

        # trans = np.mean(translation[idx_prev: idx_last], axis = 0)
        # R = np.mean(Rs[idx_prev: idx_last], axis = 0)

        # why
        trans[2] += -10
        c = -np.dot(R, trans)

        # # no why
        # c = trans

        pose = np.eye(4)
        pose[:3, :3] = R
        
        # why
        c *= 0.27
        c[1] += 0.015
        c[2] += 0.161
        # # c[2] += 0.050  # 0.160

        pose[0, 3] = c[0]
        pose[1, 3] = c[1]
        pose[2, 3] = c[2]

        # focal = 2985.29
        w = 1024 # 224
        h = 1024 # 224

        K = np.eye(3)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0

        # Rot = np.eye(3)
        # Rot[0, 0] = 1
        # Rot[1, 1] = -1
        # Rot[2, 2] = -1
        # pose[:3, :3] = np.dot(pose[:3, :3], Rot)

        # fix intrinsics
        K[0,0] = 2985.29/700 * focal / 1050
        K[1,1] = 2985.29/700 * focal / 1050
        K[0,2] = 1/2
        K[1,2] = 1/2
        assert K[0,1] == 0
        assert K[2,2] == 1
        assert K[1,0] == 0
        assert K[2,0] == 0
        assert K[2,1] == 0  

        # fix_pose_orig
        pose = np.array(pose).copy()

        # why
        pose[:3, 3] = pose[:3, 3] / 4.0 * 2.7
        # # no why
        # t_1 = np.array([-1.3651,  4.5466,  6.2646])
        # s_1 = np.array([-2.3178, -2.3715, -1.9653]) + 1
        # t_2 = np.array([-2.0536,  6.4069,  4.2269])
        # pose[:3, 3] = (pose[:3, 3] + t_1) * s_1 + t_2

        c = np.concatenate([pose.reshape(-1), K.reshape(-1)])
        c_list.append(c.astype(np.float32))

    return c_list

# video_dir = "datasets/HDTF/male/ToddYoung_0/train"
# video_dir = "datasets/HDTF/male/ToddYoung_0/test"
# video_dir = "datasets/HDTF/female/LaurenUnderwood_0/train"
# video_dir = "datasets/HDTF/female/LaurenUnderwood_0/test"
video_dir = "datasets/other/Candy/train"
# video_dir = "datasets/other/Candy/test"
# video_dir = "datasets/other/Obama_AD-NeRF/train"
# video_dir = "datasets/other/Obama_AD-NeRF/test"

# center = 0.5 # 112.
# focal = 1015. * 1024 / 224 * (300 / (102*1024/224)) / 700 # 1015.
# load_dict = {
# 	"camera_angle_x": 2 * np.arctan(center / focal),
#     "choose_video": video_dir,    
# }

img_crop_path = os.path.join(video_dir, 'videos/crop')
all_images_path = sorted([file.path for file in os.scandir(img_crop_path) if file.name.endswith('.jpg') or file.name.endswith('.png')])

file_mat = loadmat(os.path.join(video_dir, 'videos/inv_render/Candy.mat'))
angles = file_mat['euler']
translation = file_mat['trans']
focal = file_mat['focal']
angles *= -1
ids = file_mat['id']
condition = np.hstack([angles, translation, focal.repeat(angles.shape[0])[..., None]], )

angles, translations, focals = condition[:, 0:3], condition[:, 3:6], condition[:, -1]

angles = angles.copy()
angles[:, 0] *= -1
angles[:, 1] *= -1

Rs = compute_rotation(torch.from_numpy(angles.copy())).numpy()
c_list = process_camera_inv(translations.copy(), Rs, focals)
c_list = np.vstack(c_list)

# center = 0.5
# focal = 2985.29/700 * focal[0, 0] / 1050
center = 256
focal = 1100
load_dict = {
	"camera_angle_x": 2 * np.arctan(center / focal),
    "choose_video": video_dir,    
}

img_crop_path = os.path.join(video_dir, 'videos/crop')
all_images_path = sorted([file.path for file in os.scandir(img_crop_path) if file.name.endswith('.jpg') or file.name.endswith('.png')])
frames = []
for idx, c in enumerate(c_list):
    # if idx >= 500:
    #     break
    cam2world_matrix = c[:16].reshape(4, 4)
    intrinsics = c[16:25].reshape(3, 3)

    para_3dmm_path = os.path.join(video_dir, 'deep3dface', f'{idx:06d}.mat')
    base_code = get_face_3d_params(para_3dmm_path)[0]

    frames.append({
        "sub_idx": idx,
        "base_code": base_code.tolist(),
        "transform_matrix": cam2world_matrix.tolist(),
    })

load_dict["frames"] = frames

json_path = os.path.dirname(video_dir)
with open(f"{json_path}/transforms_{video_dir.split('/')[-1]}.json", 'w') as write_f:
	write_f.write(json.dumps(load_dict, indent=4, ensure_ascii=False))
    
# with open(f"{json_path}/transforms_test.json", 'w') as write_f:
# 	write_f.write(json.dumps(load_dict, indent=4, ensure_ascii=False))
