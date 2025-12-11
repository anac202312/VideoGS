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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch  # <--- [新增 1] 必須引入 torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # 原本的邏輯：如果圖片是 RGBA，取 Alpha channel作為 mask
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # 新增二：處理外部傳入的 Mask =======================================
    # 這是為了接應在 dataset_readers.py 讀到的 Mask
    if cam_info.mask is not None:
        # 1. 轉為 Tensor
        mask = torch.from_numpy(cam_info.mask)
        
        # 2. 確保維度正確 (1, H, W)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            
        # 3. 檢查是否需要縮放 (Resize)
        # 因為上面的 PILtoTorch 可能已經把 RGB 圖片縮小了，Mask 也要跟著縮
        target_h, target_w = gt_image.shape[1], gt_image.shape[2]
        
        if mask.shape[1] != target_h or mask.shape[2] != target_w:
            # interpolate 需要 (Batch, Channel, H, W) -> unsqueeze(0)
            # 使用 nearest neighbor 插值以保持 Mask 邊緣銳利 (或是用 bilinear)
            mask = mask.unsqueeze(0).float()
            mask = torch.nn.functional.interpolate(
                mask, 
                size=(target_h, target_w), 
                mode='nearest'
            )
            mask = mask.squeeze(0) # 變回 (C, H, W)

        loaded_mask = mask
    # ===================================================================

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
