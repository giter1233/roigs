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
import os
from PIL import Image
import torch

WARNED = False

def _try_load_depth_paths(source_path, image_name):
    candidates = [
        os.path.join(source_path, "depth_pro", f"{image_name}.jpg.npy"),
        os.path.join(source_path, "depth_pro", f"{image_name}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def _try_load_mask_path(source_path, image_name):
    candidates = [
        os.path.join(source_path, "pure_masks", f"{image_name}.png"),
        os.path.join(source_path, "pure_masks", f"{image_name}.jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def _compute_global_depth_radius(cam_infos, source_path):
    max_radius = None
    valid_count = 0
    for c in cam_infos:
        depth_path = _try_load_depth_paths(source_path, c.image_name)
        if depth_path is None:
            continue
        try:
            depth = np.load(depth_path)
        except Exception:
            continue
        # Optional: use stone mask to get stone-only depth statistics
        mask_path = _try_load_mask_path(source_path, c.image_name)
        if mask_path is not None:
            try:
                m = Image.open(mask_path).convert('L')
                m = (np.array(m) > 127)
            except Exception:
                m = None
        else:
            m = None

        # Robust depth selection: positive and finite
        if m is not None and m.shape[:2] == depth.shape[:2]:
            sel = m & np.isfinite(depth) & (depth > 0)
        else:
            sel = np.isfinite(depth) & (depth > 0)

        if np.count_nonzero(sel) < 10:
            continue

        r_i = float(np.median(depth[sel]))
        if np.isnan(r_i) or r_i <= 0:
            continue
        valid_count += 1
        if max_radius is None or r_i > max_radius:
            max_radius = r_i
    return max_radius, valid_count


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

    cam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

    # Build ROI mask based on global depth radius if available
    global_r = getattr(args, 'global_depth_radius', None)
    if global_r is not None:
        depth_path = _try_load_depth_paths(args.source_path, cam_info.image_name)
        if depth_path is not None:
            try:
                depth = np.load(depth_path)
                # resize depth to target resolution (W,H)
                dimg = Image.fromarray(depth.astype(np.float32))
                dimg = dimg.resize(resolution, resample=Image.NEAREST)
                d = np.array(dimg, dtype=np.float32)
                # Create ROI mask: keep pixels with depth <= global_r * (1+margin)
                margin = float(getattr(args, 'depth_margin_ratio', 0.0))
                thr = global_r * (1.0 + margin)
                roi = (np.isfinite(d) & (d > 0) & (d <= thr)).astype(np.float32)

                # AND with SAM2 mask if available
                sam_path = _try_load_mask_path(args.source_path, cam_info.image_name)
                if sam_path is not None:
                    try:
                        mimg = Image.open(sam_path).convert('L')
                        mimg = mimg.resize(resolution, resample=Image.NEAREST)
                        m = np.array(mimg, dtype=np.uint8)
                        m = (m > 127).astype(np.float32)
                        if m.shape[:2] == roi.shape[:2]:
                            roi = roi * m
                    except Exception:
                        pass

                # Shape: (H, W) from PIL output; convert to (1,H,W)
                roi_t = torch.from_numpy(roi)[None, ...]
                cam.mask = roi_t.to(args.data_device)
            except Exception:
                pass
    return cam


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    # Pre-compute a global depth radius (max over all views)
    if getattr(args, 'global_depth_radius', None) is None and hasattr(args, 'source_path'):
        max_r, valid = _compute_global_depth_radius(cam_infos, args.source_path)
        if max_r is not None:
            setattr(args, 'global_depth_radius', float(max_r))
            if getattr(args, 'depth_margin_ratio', None) is None:
                # default margin ratio if user didn't specify
                setattr(args, 'depth_margin_ratio', 0.0)
            print(f"[ROI] Global depth radius (max over {valid} valid views): {args.global_depth_radius:.4f} (margin ratio={getattr(args, 'depth_margin_ratio')})")
        else:
            print("[ROI] No valid depth maps found; proceed without ROI masking.")

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
