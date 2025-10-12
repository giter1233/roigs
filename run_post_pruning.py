import os
import argparse
import torch

from arguments import ModelParams, OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.system_utils import mkdir_p, searchForMaxIteration


def load_cfg(model_path: str):
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}. Make sure you pass --model_path from a finished training run.")
    with open(cfg_path, "r") as f:
        txt = f.read().strip()
    try:
        if txt.startswith("Namespace("):
            return eval(txt, {"__builtins__": {}}, {"Namespace": argparse.Namespace})
        else:
            import ast
            return ast.literal_eval(txt)
    except Exception as e:
        raise ValueError(f"Failed to parse cfg_args at {cfg_path}: {e}")


def resolve_iteration(model_path: str, iteration: int) -> int:
    pc_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        return searchForMaxIteration(pc_dir)
    return iteration


def next_free_iteration(model_path: str, start_iter: int) -> int:
    pc_dir = os.path.join(model_path, "point_cloud")
    it = start_iter + 1
    while os.path.isdir(os.path.join(pc_dir, f"iteration_{it}")):
        it += 1
    return it


def main():
    parser = argparse.ArgumentParser(description="Post-training pruning for Gaussian Splatting models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the training output folder (the one containing cfg_args and point_cloud/")
    parser.add_argument("--iteration", type=int, default=-1, help="Which iteration to load (-1 = latest)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--min_opacity", type=float, help="Absolute opacity threshold. Prune gaussians with opacity < threshold")
    group.add_argument("--percentile", type=float, help="Quantile q in [0,1]. Prune gaussians below the q-quantile of opacity")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use")

    args = parser.parse_args()

    torch.cuda.set_device(torch.device("cuda:0")) if args.device.startswith("cuda") else None

    # Load saved training config to recover SH degree and optimizer settings
    saved_cfg = load_cfg(args.model_path)

    # Prepare param groups (we reuse the repo's argument system to build training args)
    tmp_parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(tmp_parser)
    op = OptimizationParams(tmp_parser)
    # Build group params from saved namespace
    lp_group = lp.extract(saved_cfg)
    op_group = op.extract(saved_cfg)
    
    # Ensure all required attributes are present
    if not hasattr(op_group, 'percent_dense'):
        op_group.percent_dense = 0.01
    if not hasattr(op_group, 'position_lr_init'):
        op_group.position_lr_init = 0.00004
    if not hasattr(op_group, 'position_lr_final'):
        op_group.position_lr_final = 0.000002
    if not hasattr(op_group, 'position_lr_delay_mult'):
        op_group.position_lr_delay_mult = 0.01
    if not hasattr(op_group, 'position_lr_max_steps'):
        op_group.position_lr_max_steps = 30000
    if not hasattr(op_group, 'feature_lr'):
        op_group.feature_lr = 0.0025
    if not hasattr(op_group, 'shfeature_lr'):
        op_group.shfeature_lr = 0.005
    if not hasattr(op_group, 'opacity_lr'):
        op_group.opacity_lr = 0.025
    if not hasattr(op_group, 'scaling_lr'):
        op_group.scaling_lr = 0.005
    if not hasattr(op_group, 'rotation_lr'):
        op_group.rotation_lr = 0.001

    # Build Gaussian model and load PLY
    gaussians = GaussianModel(lp_group.sh_degree, getattr(op_group, "optimizer_type", "default"))

    load_iter = resolve_iteration(args.model_path, args.iteration)
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{load_iter}", "point_cloud.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY not found at {ply_path}")

    gaussians.load_ply(ply_path)
    gaussians.training_setup(op_group)

    before = gaussians.get_xyz.shape[0]

    if args.percentile is not None:
        if not (0.0 <= args.percentile <= 1.0):
            raise ValueError("--percentile must be within [0,1]")
        gaussians.only_prune(args.percentile, percent=True)
    else:
        if not (0.0 <= args.min_opacity <= 1.0):
            raise ValueError("--min_opacity must be within [0,1]")
        gaussians.only_prune(args.min_opacity, percent=False)

    after = gaussians.get_xyz.shape[0]
    kept_ratio = after / max(1, before)
    print(f"Pruned gaussians: {before - after} removed, {after} kept ({kept_ratio:.2%}).")

    # Save to a new numeric iteration folder to stay compatible with loaders
    new_iter = next_free_iteration(args.model_path, load_iter)
    out_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{new_iter}")
    mkdir_p(out_dir)
    out_ply = os.path.join(out_dir, "point_cloud.ply")
    gaussians.save_ply(out_ply)

    print(f"Saved pruned model to: {out_ply}")
    print(f"Tip: load with load_iteration={new_iter} or keep using -1 to pick the latest.")


if __name__ == "__main__":
    main()