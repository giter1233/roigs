# precompute_depths.py - 预处理脚本：为数据集生成 Depth Pro 深度图
import os
import torch
import depth_pro
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

def main(source_path):
    model, transform = depth_pro.create_model_and_transforms()
    model.eval().cuda()

    # 输入输出路径
    image_dir = os.path.join(source_path, 'images')
    depth_dir = os.path.join(source_path, 'depth_pro')
    os.makedirs(depth_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    # 批量处理
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # 加载图像和焦距
            image, _, f_px = depth_pro.load_rgb(img_path)
            image = transform(image).unsqueeze(0).cuda()
            
            # 推理
            with torch.no_grad():
                prediction = model.infer(image, f_px=f_px)
            
            # 保存结果
            rel_path = os.path.relpath(img_path, image_dir)
            save_path = os.path.join(depth_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存深度图和元数据
            np.save(save_path + '.npy', prediction["depth"].cpu().numpy())  # 添加.cpu()
            np.savez(save_path + '_meta.npz', focallength_px=prediction["focallength_px"].cpu().numpy())  # 同样需要转换
            
            # 可选：保存PNG预览图
            depth_np = prediction["depth"].cpu().numpy()
            depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
            Image.fromarray((depth_normalized * 255).astype(np.uint8)).save(save_path + '.png')

        except Exception as e:
            print(f"处理 {img_path} 失败: {str(e)}")  
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute Depth Pro depths")
    parser.add_argument('-s', '--source_path', type=str, required=True, help="Path to dataset (e.g., data/bonsai)")
    args = parser.parse_args()
    main(args.source_path)