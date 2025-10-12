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
import math

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = lambda x: torch.nn.functional.normalize(x, dim=-1)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        shopt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        sh_l = [{'params': [self._features_rest], 'lr': training_args.shfeature_lr / 20.0, "name": "f_rest"}]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # Total number of points
        N = self._xyz.shape[0]

        # Prepare property names following the same order as construct_list_of_attributes
        prop_names = self.construct_list_of_attributes()

        # Write binary little-endian PLY header
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            "comment saved in streaming mode to reduce memory peak",
            f"element vertex {N}",
        ]
        for name in prop_names:
            header_lines.append(f"property float {name}")
        header_lines.append("end_header")
        header = ("\n".join(header_lines) + "\n").encode("ascii")

        # Determine per-vertex attribute length (number of floats)
        # 3 (xyz) + 3 (normals) + f_dc + f_rest + 1 (opacity) + scales + rotations
        # We'll compute these from tensor shapes
        f_dc_cols = (self._features_dc.shape[1] * self._features_dc.shape[2])  # 3 * 1 = 3
        f_rest_cols = (self._features_rest.shape[1] * self._features_rest.shape[2])
        scale_cols = self._scaling.shape[1]
        rot_cols = self._rotation.shape[1]
        floats_per_vertex = 3 + 3 + f_dc_cols + f_rest_cols + 1 + scale_cols + rot_cols

        # Set a memory budget for each chunk (in bytes) to control peak usage
        # 64 MB per chunk by default
        mem_budget_bytes = 64 * 1024 * 1024
        # Compute chunk size based on attribute footprint
        # Ensure at least 1 vertex per chunk
        chunk_size = max(1, min(int(mem_budget_bytes // (floats_per_vertex * 4)), int(N)))

        with open(path, "wb") as f:
            f.write(header)

            start = 0
            while start < N:
                end = min(start + chunk_size, N)
                # Fetch data for this chunk on CPU with minimal intermediate allocations
                xyz = self._xyz[start:end].detach().to("cpu").numpy()  # (M, 3)
                normals = np.zeros_like(xyz, dtype=np.float32)  # (M, 3)

                f_dc = (
                    self._features_dc[start:end]
                    .detach()
                    .transpose(1, 2)
                    .flatten(start_dim=1)
                    .contiguous()
                    .to("cpu")
                    .numpy()
                )  # (M, f_dc_cols)
                f_rest = (
                    self._features_rest[start:end]
                    .detach()
                    .transpose(1, 2)
                    .flatten(start_dim=1)
                    .contiguous()
                    .to("cpu")
                    .numpy()
                )  # (M, f_rest_cols)
                opacities = self._opacity[start:end].detach().to("cpu").numpy()  # (M, 1)
                scales = self._scaling[start:end].detach().to("cpu").numpy()  # (M, scale_cols)
                rots = self._rotation[start:end].detach().to("cpu").numpy()  # (M, rot_cols)

                # Concatenate per-vertex attributes in the exact order expected by load_ply
                attributes = np.concatenate(
                    (xyz, normals, f_dc, f_rest, opacities, scales, rots), axis=1
                ).astype("<f4", copy=False)

                # Stream write this chunk as binary little-endian
                f.write(attributes.tobytes(order="C"))

                start = end


    def reset_opacity(self, min_opacity):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*min_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def only_prune(self, min_opacity, percent=False):
        if percent is True:
            opacity_array = self.get_opacity.detach().flatten()
            q = min_opacity
            min_opacity = torch.quantile(opacity_array, q)
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()

        valid_points_mask = ~prune_mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        torch.cuda.empty_cache()

    def densify_and_prune_Improved(self, scores, min_opacity, budget, opt, iteration, mask=None, camera=None):
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0

        min_grad = opt.densify_grad_threshold

        if scores is None or iteration > 14500:
            scores = grad_vars.squeeze()
            if self.get_opacity.shape[0] < budget and iteration > 14500:
                min_grad = min_grad / 1.5
        # ensure scores on correct device for later masking/sampling
        scores = scores.to(self.get_xyz.device)

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= min_grad, True, False)
        
        # 严格 ROI 模式：如提供 mask 和 camera，则将 3D 点投影到当前相机图像坐标系，
        # 仅允许 ROI 内的点参与 densify，ROI 外的点不再生长。
        if mask is not None and camera is not None:
            try:
                # 准备相机与图像尺寸
                W = int(camera.image_width)
                H = int(camera.image_height)
                full = camera.full_proj_transform  # (4,4)

                # 统一 mask 到二维 (H, W)，阈值化为 bool
                m = mask
                if m.dim() == 3:
                    # 形状可能是 (1,H,W) 或 (3,H,W)
                    if m.shape[0] == 1:
                        m2d = m[0]
                    else:
                        m2d = m.float().mean(dim=0)
                elif m.dim() == 2:
                    m2d = m
                else:
                    # 不支持的形状，回退为全 True
                    m2d = torch.ones((H, W), device=m.device, dtype=torch.float32)
                m2d = (m2d > 0.5)

                xyz = self.get_xyz  # (N,3)
                N = xyz.shape[0]
                ones = torch.ones((N, 1), device=xyz.device, dtype=xyz.dtype)
                xyz_h = torch.cat([xyz, ones], dim=1)  # (N,4)

                # 世界->裁剪坐标，采用 row-vector 右乘： (N,4) @ (4,4) -> (N,4)
                clip = xyz_h @ full  # (N,4)
                w = clip[:, 3:4]
                # 过滤 w<=0 的点（在投影后无效）
                valid_w = (w.squeeze(-1) > 0)
                ndc = clip[:, :3] / (w + 1e-8)

                # NDC [-1,1] -> 像素坐标 [0, W-1] / [0, H-1]
                u = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)
                # y 轴翻转，适配图像以左上角为原点
                v = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (H - 1)
                z_ok = (ndc[:, 2] >= 0.0) & (ndc[:, 2] <= 1.0)

                # 在图像范围内
                in_x = (u >= 0.0) & (u <= (W - 1))
                in_y = (v >= 0.0) & (v <= (H - 1))
                in_img = in_x & in_y

                # 采样 mask
                u_l = u.round().long().clamp_(0, W - 1)
                v_l = v.round().long().clamp_(0, H - 1)
                roi_vals = m2d[v_l, u_l]
                roi_ok = roi_vals.bool()

                # 深度门控：限制非常远的点参与 densify，避免远景“炸开”
                # 默认阈值 0.9；如提供 depth_margin_ratio>0，则稍微放宽/收紧阈值
                ratio = getattr(opt, 'depth_margin_ratio', 0.0)
                z_gate = 0.9 if not isinstance(ratio, (float, int)) else float(max(0.0, min(1.0, 0.9 - 0.3 * ratio)))
                depth_ok = (ndc[:, 2] <= z_gate)

                # 组合有效性：必须在前方、在图像内、且落在 ROI 内、并通过深度门控
                gating_mask = valid_w & z_ok & in_img & roi_ok & depth_ok

                # 用 gating 严格筛选densify候选
                grad_qualifiers = grad_qualifiers & gating_mask
            except Exception as e:
                # 如投影失败，保持原有行为
                pass
        
        # Fallback：若没有任何点通过阈值/门控，则退回到基于得分的选择，避免“完全不增长”
        if torch.sum(grad_qualifiers) == 0:
            score_mask = (scores > 0)
            if torch.sum(score_mask) == 0:
                # 仍然全为0，则取 top-k 作为候选
                curr_n = self.get_xyz.shape[0]
                k = min(max(64, int(0.005 * curr_n)), scores.numel())
                if k > 0:
                    top_idx = torch.topk(scores, k, sorted=False).indices
                    grad_qualifiers = torch.zeros_like(grad_qualifiers, device=grad_qualifiers.device)
                    grad_qualifiers[top_idx] = True
            else:
                grad_qualifiers = score_mask
        
        total_sum = torch.sum(grad_qualifiers).item()
        curr_points = len(self.get_xyz)
        budget = min(budget, total_sum + curr_points)

        all_budget = budget - curr_points
        # Limit per-step densification based on profile tiers; explicit user cap overrides
        per_step_cap_user = getattr(opt, 'max_new_points_per_iter', None)
        profile = str(getattr(opt, 'densify_cap_profile', 'ultra')).lower()
        # Define heuristic tiers: (ratio, min_cap, max_cap)
        tiers = {
            'ultra':        (0.020, 2048, 32768),
            'ultra+':       (0.040, 4096, 65536),
        }
        if profile not in tiers:
            profile = 'ultra'
        ratio, cap_min, cap_max = tiers[profile]
        heuristic_cap = int(max(cap_min, min(cap_max, ratio * curr_points)))
        allowed_new = min(all_budget, heuristic_cap)
        if per_step_cap_user is not None:
            try:
                allowed_new = min(all_budget, int(per_step_cap_user))
            except Exception:
                pass

        if allowed_new > 0:
            self.long_axis_split(scores.clone(), allowed_new, grad_qualifiers, opt.split_distance, opt.opacity_reduction)

        # 基础 prune mask：按不透明度（仅在使用 mask 时执行）
        if mask is not None:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        else:
            prune_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        
        # 如果启用严格 ROI（mask+camera），对 ROI 外及极远处采用更严格的 prune 策略
        if mask is not None and camera is not None and iteration < 14900:
            try:
                # 复用上面的投影（如前面异常，这里再算一次）
                W = int(camera.image_width)
                H = int(camera.image_height)
                full = camera.full_proj_transform

                m = mask
                if m.dim() == 3:
                    if m.shape[0] == 1:
                        m2d = m[0]
                    else:
                        m2d = m.float().mean(dim=0)
                elif m.dim() == 2:
                    m2d = m
                else:
                    m2d = torch.ones((H, W), device=m.device, dtype=torch.float32)
                m2d = (m2d > 0.5)

                xyz = self.get_xyz
                N = xyz.shape[0]
                ones = torch.ones((N, 1), device=xyz.device, dtype=xyz.dtype)
                xyz_h = torch.cat([xyz, ones], dim=1)
                clip = xyz_h @ full
                w = clip[:, 3:4]
                valid_w = (w.squeeze(-1) > 0)
                ndc = clip[:, :3] / (w + 1e-8)
                u = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)
                v = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (H - 1)
                z_ok = (ndc[:, 2] >= 0.0) & (ndc[:, 2] <= 1.0)
                in_x = (u >= 0.0) & (u <= (W - 1))
                in_y = (v >= 0.0) & (v <= (H - 1))
                in_img = in_x & in_y

                u_l = u.round().long().clamp_(0, W - 1)
                v_l = v.round().long().clamp_(0, H - 1)
                roi_vals = m2d[v_l, u_l]
                roi_ok = roi_vals.bool()
                outside_roi = valid_w & z_ok & in_img & (~roi_ok)

                # 极远处（深度接近远裁剪面）的额外裁剪
                far_mask = valid_w & z_ok & in_img & (ndc[:, 2] > 0.95)

                # 在 ROI 边缘附近构造一条窄带，抑制边缘处高斯过亮过大
                try:
                    import torch.nn.functional as F
                    m2d_f = m2d.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                    dil = F.max_pool2d(m2d_f, kernel_size=3, stride=1, padding=1)
                    inv = 1.0 - m2d_f
                    ero = 1.0 - F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
                    edge_band = (dil > 0.5) & (ero < 0.5)  # 约1像素宽边带
                    edge_vals = edge_band[0, 0, v_l, u_l]
                    inside_roi_edge = valid_w & z_ok & in_img & roi_ok & edge_vals

                    # 对 ROI 内边缘带做不透明度衰减与上限，避免边缘高亮“大饼”
                    if inside_roi_edge.any():
                        curr_opacity = self.get_opacity.clone()  # (N,1)
                        cap = 0.75
                        damp = 0.97
                        curr_opacity[inside_roi_edge, 0] = torch.clamp(curr_opacity[inside_roi_edge, 0] * damp, max=cap)
                        new_opacity_param = self.inverse_opacity_activation(curr_opacity)
                        optimizable_tensors = self.replace_tensor_to_optimizer(new_opacity_param, "opacity")
                        self._opacity = optimizable_tensors["opacity"]
                except Exception:
                    pass

                # 对 ROI 外和极远处提高裁剪阈值（更激进地移除）
                adaptive_min_opacity = min_opacity + 0.02
                adaptive_prune_mask = (self.get_opacity.squeeze() < adaptive_min_opacity)
                prune_mask = torch.logical_or(prune_mask, (adaptive_prune_mask & (outside_roi | far_mask)))
            except Exception as e:
                pass
        
        # Lightweight ROI-aware pruning during densification
        if iteration < 14900:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_abs(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def long_axis_split(self, grads, budget, filter, split_distance, opacity_reduction):
        grads[~filter] = 0
        n_init_points = self.get_xyz.shape[0]

        padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
        padded_importance[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")

        num = (padded_importance > 0).sum().item()
        if budget > num:
            budget = num

        sampled_indices = torch.multinomial(padded_importance, budget, replacement=False)
        selected_pts_mask[sampled_indices] = True
        stds = self.get_scaling[selected_pts_mask]

        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)

        samples = stds * mask * 3

        reduction = opacity_reduction
        rate = split_distance
        x1 = samples * rate
        rate_w = 1 - rate
        rate_h = math.sqrt(1-rate*rate)
        x1 = torch.cat([x1, -x1], dim=0)

        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        new_xyz = torch.bmm(rots, x1.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(2, 1)
        new_scaling = self.scaling_inverse_activation(stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1) * rate_h)
        new_opacity = inverse_sigmoid(self.get_opacity[selected_pts_mask] * reduction).repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
