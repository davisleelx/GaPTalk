import os
import json
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

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
        ones = np.ones([batch_size, 1])
        zeros = np.zeros([batch_size, 1])
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = np.concatenate([
            ones, zeros, zeros,
            zeros, np.cos(x), -np.sin(x), 
            zeros, np.sin(x), np.cos(x)
        ], axis=1).reshape([batch_size, 3, 3])
        
        rot_y = np.concatenate([
            np.cos(y), zeros, np.sin(y),
            zeros, ones, zeros,
            -np.sin(y), zeros, np.cos(y)
        ], axis=1).reshape([batch_size, 3, 3])

        rot_z = np.concatenate([
            np.cos(z), -np.sin(z), zeros,
            np.sin(z), np.cos(z), zeros,
            zeros, zeros, ones
        ], axis=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.transpose(0, 2, 1)

center = 0.5 # 112.
focal = 1015. * 1024 / 224 * (300 / (102*1024/224)) / 700 # 1015.

# video_dir = "datasets/HDTF/female/HaleyStevens_0/train"
# video_dir = "datasets/HDTF/female/HaleyStevens_0/test"
# video_dir = "datasets/HDTF/female/KirstenGillibrand_0/train"
# video_dir = "datasets/HDTF/female/KirstenGillibrand_0/test"
# video_dir = "datasets/HDTF/female/LaurenUnderwood_0/train"
# video_dir = "datasets/HDTF/female/LaurenUnderwood_0/test"
# video_dir = "datasets/HDTF/female/TinaSmith_0/train"
# video_dir = "datasets/HDTF/female/TinaSmith_0/test"

# video_dir = "datasets/HDTF/male/AndyKim_0/train"
# video_dir = "datasets/HDTF/male/AndyKim_0/test"
# video_dir = "datasets/HDTF/male/ChrisMurphy0_0/train"
# video_dir = "datasets/HDTF/male/ChrisMurphy0_0/test"
# video_dir = "datasets/HDTF/male/JohnSarbanes0_0/train"
video_dir = "datasets/HDTF/male/JohnSarbanes0_0/test"
# video_dir = "datasets/HDTF/male/ToddYoung_0/train"
# video_dir = "datasets/HDTF/male/ToddYoung_0/test"

# video_dir = "datasets/other/Candy/train"
# video_dir = "datasets/other/Candy/test"
# video_dir = "datasets/other/Obama_AD-NeRF/train"
# video_dir = "datasets/other/Obama_AD-NeRF/test"

load_dict = {
	"camera_angle_x": 2 * np.arctan(center / focal),
    "choose_video": video_dir,    
}

img_crop_path = os.path.join(video_dir, 'videos/crop')
all_images_path = sorted([file.path for file in os.scandir(img_crop_path) if file.name.endswith('.jpg') or file.name.endswith('.png')])

frames = []
for idx, image_path in tqdm(enumerate(all_images_path)):
    # if idx >= 500:
    #     break
    file_path = image_path[:-4]
    
    para_3dmm_path = os.path.join(video_dir, 'deep3dface', f'{idx:06d}.mat')
    base_code = get_face_3d_params(para_3dmm_path)[0]

    base_w2c_Tvec = base_code[254:257]
    base_w2c_Tvec[-1] -= 10
    base_c2w_Rmat = compute_rotation(base_code[None, 224:227]).squeeze(0)
    base_c2w_Tvec = -(np.matmul(base_c2w_Rmat, base_w2c_Tvec.reshape(3, 1)))
    base_c2w_Tvec = base_c2w_Tvec.reshape(3, 1)

    # Rot = np.eye(3)
    # Rot[0, 0] = 1
    # Rot[1, 1] = -1
    # Rot[2, 2] = -1
    # base_c2w_Rmat = np.matmul(base_c2w_Rmat, Rot)
    
    transform_matrix = np.zeros((4, 4))
    transform_matrix[:3, :3] = base_c2w_Rmat
    transform_matrix[:3, 3] = base_c2w_Tvec.squeeze()
    transform_matrix[3, 3] = 1

    frames.append({
        "sub_idx": idx,
        "base_code": base_code.tolist(),
        "transform_matrix": transform_matrix.tolist(),
    })

load_dict["frames"] = frames

json_path = os.path.dirname(video_dir)
with open(f"{json_path}/transforms_{video_dir.split('/')[-1]}.json", 'w') as write_f:
	write_f.write(json.dumps(load_dict, indent=4, ensure_ascii=False))
    
# with open(f"{json_path}/transforms_test.json", 'w') as write_f:
# 	write_f.write(json.dumps(load_dict, indent=4, ensure_ascii=False))
