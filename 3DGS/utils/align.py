import torch
import torch.nn.functional as F

def rescale_mask(input_mask, transform_params: list):
    """
    Uncrops and rescales (i.e., resizes) the given scaled and cropped mask back to the
    resolution of the original image using the given transformation parameters.
    input_mask: (B, 3, H, W)
    """
    B, C, H, W = input_mask.shape
    input_mask = F.interpolate(input_mask, (512, 512), mode='bilinear')
    # scale_w = W / transform_params[:, 0]
    # scale_h = H / transform_params[:, 1]

    target_size = 1024. # * scale_w
    center_crop_size = 700 # * scale_w

    device = input_mask.device
    original_image_width, original_image_height = transform_params[:, 0].to(torch.int32), transform_params[:, 1].to(torch.int32)
    # original_image_width, original_image_height = W, H
    s = transform_params[:, 2]
    t = transform_params[:, 3:] # * torch.stack([scale_w, scale_h], dim=1) # (2)

    scaled_image_w = (original_image_width * s).to(torch.int32)
    scaled_image_h = (original_image_height * s).to(torch.int32)
    left = (scaled_image_w/2 - target_size/2 + (t[:, 0] - original_image_width/2)*s).to(torch.int32)
    up = (scaled_image_h/2 - target_size/2 + (original_image_height/2 - t[:, 1])*s).to(torch.int32)

    # Parse transform params.
    mask_scaled_list = []
    for i in range(B):
        mask_scaled_0 = torch.ones([C, int(target_size), int(target_size)]).to(device)
        x = F.interpolate(input_mask[[i]], (center_crop_size, center_crop_size), mode='bilinear')
        mask_scaled_0[:, int(target_size-center_crop_size)//2:int(target_size+center_crop_size)//2, int(target_size-center_crop_size)//2:int(target_size+center_crop_size)//2] = x
        x = mask_scaled_0

        if left[i] < 0:
            left_scale = 0
            right_scale = int(left_scale + min(target_size + left[i], scaled_image_w[i]))
        else:
            left_scale = left[i]
            right_scale = int(left_scale + min(target_size, scaled_image_w[i] - left[i]))
        if up[i] < 0:
            up_scale = 0
            below_sclae = int(up_scale + min(target_size + up[i], scaled_image_h[i]))
        else:
            up_scale = up[i]
            below_sclae = int(up_scale + min(target_size, scaled_image_h[i] - up[i]))

        if left[i] < 0:
            left_target = -left[i]
            right_target = int(left_target + min(target_size - left_target, scaled_image_w[i]))
        else:
            left_target = 0
            right_target = int(left_target + min(target_size, scaled_image_w[i] - left[i]))
        if up[i] < 0:
            up_target = -up[i]
            below_target = int(up_target + min(target_size - up_target, scaled_image_h[i]))
        else:
            up_target = 0
            below_target = int(up_target + min(target_size, scaled_image_h[i] - up[i]))

        mask_scaled = torch.ones([C, scaled_image_w[i], scaled_image_h[i]]).to(device)
        mask_scaled[:, up_scale:below_sclae, left_scale:right_scale] = \
            x[:, up_target:below_target, left_target:right_target]
        # Rescale the uncropped mask back to the resolution of the original image.
        uncropped_and_rescaled_mask = F.interpolate(mask_scaled.unsqueeze(0), (original_image_width[i], original_image_height[i]), mode='bilinear')
        # uncropped_and_rescaled_mask[uncropped_and_rescaled_mask==1] = self.img_ori_tensor[[i]][uncropped_and_rescaled_mask==1]
        mask_scaled_list.append(uncropped_and_rescaled_mask)
    
    uncropped_and_rescaled_mask = torch.cat(mask_scaled_list, dim=0)
    
    uncropped_and_rescaled_mask = F.interpolate(uncropped_and_rescaled_mask, (H, W), mode='bilinear')

    return uncropped_and_rescaled_mask
