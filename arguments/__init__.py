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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"  # "cpu"
        self.eval = True
        # 深度半径mask的裕量比例（方案A），例如0.05表示阈值=global_r*(1+0.05)
        self.depth_margin_ratio = 0.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00004
        self.position_lr_final = 0.000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.shfeature_lr = 0.005
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0003
        self.random_background = False
        self.optimizer_type = "default"

        self.budget = 300_0000
        self.split_distance = 0.45
        self.opacity_reduction = 0.6
        self.lambda_consistency = 0.03
        
        # LPDR (Laplacian Pyramid Dynamic Loss) Parameters
        self.lpdr_weight = 0.2
        self.lpdr_levels = 3
        self.lpdr_kernel_size = 5
        self.lpdr_sigma = 1.0
        self.lpdr_weight_gamma = 1.2
        self.lpdr_include_dc = True
        self.lpdr_dc_weight = 0.3
        
        # Mask training parameters
        self.use_mask = False
        # ROI scheduling for densify/prune and loss masking
        # 1) 从较晚的迭代起，对致密化/剪枝启用严格ROI门控（传入mask&camera）
        self.roi_grow_from_iter = 10_000
        # 2) 在该迭代之前，对ROI外/远距点采用更激进的剪枝；到期后放松（只按不透明度/默认策略）
        self.roi_prune_strict_until = 15_000
        # 3) 从较晚的迭代起，训练损失聚焦ROI：将ROI外像素的损失权重降为很小
        self.roi_loss_focus_after = 20_000
        # ROI损失mask外区域的权重（0完全不计，建议0.02~0.1）
        self.roi_loss_outside_weight = 0.05
        # 训练时对mask进行膨胀像素数（缓解边界误差），设置为2像素
        self.roi_mask_dilate = 2
        # 开启软边缘（池化/模糊）
        self.roi_mask_soft_edges = True
        # 软边缘模糊核大小（奇数）
        self.roi_mask_blur_kernel = 5
        # 新增：ROI门控方式（always/late），以及在启用ROI时的全局不透明度阈值
        self.roi_gate_mode = "always"
        self.min_opacity_thr_when_mask = 0.0
        # 单次致密化启发式上限的策略档位：conservative/balanced/aggressive/ultra
        self.densify_cap_profile = "ultra"
        
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
