import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 基础配置
class Config:
    # 模型配置
    MODEL_PATH = "checkpoints/sam2.1_hiera_large.pt"  # SAM2模型路径
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"                # 模型对应配置文件
    USE_GPU = True                                 # 有NVIDIA显卡设为True
    masked_save_dir = "../data/shibei/masked_images"  # 新增masked目录
    pure_mask_dir = "../data/shibei/pure_masks"
    IMAGE_SEQ_DIR = "../data/shibei/images"    # 图片存放目录
    SEG_SAVE_DIR = "../data/shibei/segmenteds"  # 分割结果图片保存目录
    CROP_SAVE_DIR = "../data/shibei/cropped/images"  
    # 单目标配置
    SINGLE_TARGET = {
        1: "target"  # 替换为你的目标
    }
    ANNOTATE_IMG_IDX = 0  # 在第1张图片（索引0）上标注
    VIS_STRIDE = 60       # 预览

# 交互式标注类
class InteractiveAnnotator:
    def __init__(self, image):
        self.image = image
        self.coords = []
        self.labels = []
        self.fig = None
        self.ax = None
        
    def onclick(self, event):
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # 左键：前景点（标签1）
        if event.button == 1:
            self.coords.append([x, y])
            self.labels.append(1)
            self.ax.plot(x, y, 'go', markersize=8, markeredgecolor='white', markeredgewidth=2)
            print(f" 添加前景点: ({x}, {y})")
            
        # 右键：背景点（标签0）
        elif event.button == 3:
            self.coords.append([x, y])
            self.labels.append(0)
            self.ax.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
            print(f" 添加背景点: ({x}, {y})")
            
        self.fig.canvas.draw()
        
    def onkey(self, event):
        if event.key == 'u' and len(self.coords) > 0:  # u键撤销最后一个点
            removed_coord = self.coords.pop()
            removed_label = self.labels.pop()
            print(f"↶ 撤销点: {removed_coord} (标签: {removed_label})")
            self.redraw_points()
            
        elif event.key == 'c':  # c键清除所有点
            self.coords.clear()
            self.labels.clear()
            print("🗑️ 清除所有标注点")
            self.redraw_points()
            
        elif event.key == 'enter' or event.key == ' ':  # 回车或空格键完成标注
            if len(self.coords) > 0:
                plt.close(self.fig)
                print(f"标注完成！共标注 {len(self.coords)} 个点")
            else:
                print(" 请至少标注一个点")
                
    def redraw_points(self):
        """重新绘制所有标注点"""
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(
            "交互式标注 - 左键:前景点(绿) | 右键:背景点(红) | U:撤销 | C:清除 | 回车:完成",
            fontsize=12
        )
        
        for i, (coord, label) in enumerate(zip(self.coords, self.labels)):
            x, y = coord
            if label == 1:
                self.ax.plot(x, y, 'go', markersize=8, markeredgecolor='white', markeredgewidth=2)
            else:
                self.ax.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
                
        self.fig.canvas.draw()
        
    def start_annotation(self):
        """开始交互式标注"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title(
            "交互式标注 - 左键:前景点(绿) | 右键:背景点(红) | U:撤销 | C:清除 | 回车:完成",
            fontsize=12
        )
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        
        print("\n🖱️ 交互式标注说明:")
        print("   左键点击: 添加前景点（绿色圆点）")
        print("   右键点击: 添加背景点（红色圆点）")
        print("   U键: 撤销最后一个点")
        print("   C键: 清除所有点")
        print("   回车/空格: 完成标注")
        print("\n请在图片上点击标注目标...")
        
        plt.show()
        
        if len(self.coords) > 0:
            return np.array(self.coords, dtype=np.float32), np.array(self.labels, dtype=np.int32)
        else:
            return None, None

# 工具函数
def init_sam2_device(cfg):
    """初始化GPU/CPU设备"""
    if cfg.USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"✅ 使用GPU设备：{torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("ℹ️  使用CPU设备")
    return device

def show_mask(mask, ax, alpha=0.6):
    """显示单目标分割掩码（固定蓝色）"""
    # 确保掩码是2D数组
    if mask.ndim > 2:
        mask_2d = mask.squeeze()
    else:
        mask_2d = mask
    
    color = np.array([30/255, 144/255, 255/255, alpha])
    h, w = mask_2d.shape[-2:]
    mask_image = mask_2d.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """显示标注点（前景绿色，背景红色）"""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def add_mask_to_image(image, mask, obj_name, alpha=0.4):
    """将掩码叠加到原图上"""
    # 转换为numpy数组
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # 确保掩码是2D数组
    if mask.ndim > 2:
        mask_2d = mask.squeeze()
    else:
        mask_2d = mask
    
    # 创建彩色掩码（蓝色）
    color_mask = np.zeros_like(image_np)
    color_mask[:, :, 2] = 255  # 蓝色通道
    
    # 应用掩码
    masked_image = image_np.copy()
    mask_area = mask_2d > 0
    masked_image[mask_area] = (1 - alpha) * image_np[mask_area] + alpha * color_mask[mask_area]
    
    return Image.fromarray(masked_image.astype(np.uint8))

def crop_object_from_image(image, mask, obj_name):
    """从图像中裁剪出目标对象"""
    # 转换为numpy数组
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # 确保掩码是2D数组
    if mask.ndim > 2:
        mask_2d = mask.squeeze()
    else:
        mask_2d = mask
    
    # 找到掩码的边界框
    mask_indices = np.where(mask_2d > 0)
    if len(mask_indices[0]) == 0:
        return None
    
    y_min, y_max = mask_indices[0].min(), mask_indices[0].max()
    x_min, x_max = mask_indices[1].min(), mask_indices[1].max()
    
    # 裁剪图像
    cropped_image = image_np[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask_2d[y_min:y_max+1, x_min:x_max+1]
    
    # 应用掩码（背景设为透明或白色）
    if cropped_image.shape[2] == 3:  # RGB图像
        # 创建RGBA图像
        rgba_image = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = cropped_image
        rgba_image[:, :, 3] = (cropped_mask > 0) * 255  # Alpha通道
        return Image.fromarray(rgba_image)
    
    return Image.fromarray(cropped_image)

def crop_masked_from_image(image, mask, obj_name):
    """从图像中裁剪出带掩码的目标对象（背景透明）"""
    # 转换为numpy数组
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # 确保掩码是2D数组
    if mask.ndim > 2:
        mask_2d = mask.squeeze()
    else:
        mask_2d = mask
    
    # 创建RGBA图像
    rgba_image = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = image_np
    rgba_image[:, :, 3] = (mask_2d > 0) * 255  # Alpha通道：掩码区域不透明，其他区域透明
    
    return Image.fromarray(rgba_image)

def interactive_segmentation():
    """交互式分割主函数"""
    # 初始化环境与模型
    print(f"初始化环境（目标：{Config.SINGLE_TARGET[1]} | 交互式标注模式）")

    # 检查模型文件
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"模型不存在：{Config.MODEL_PATH}\n"
        )
    
    # 初始化设备
    init_sam2_device(Config)
    
    # 加载SAM2模型
    try:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(Config.MODEL_CFG, Config.MODEL_PATH)
        print("✅ SAM2模型加载成功！")
    except ImportError as e:
        raise ImportError(f" 加载SAM2失败：{e}\n请先在SAM2源码根目录执行：pip install --no-build-isolation -e .")

    # 读取图片序列
    print(f"2. 读取图片序列（目录：{Config.IMAGE_SEQ_DIR}）")
    
    # 检查图片目录
    if not os.path.exists(Config.IMAGE_SEQ_DIR):
        raise FileNotFoundError(f" 图片目录不存在：{Config.IMAGE_SEQ_DIR}\n请将所有图片放入该目录")
    
    # 读取图片文件（仅保留常见格式）
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
    image_names = [
        f for f in os.listdir(Config.IMAGE_SEQ_DIR)
        if f.lower().endswith(image_extensions)
    ]
    
    # 按文件名中的数字排序（确保与拍摄顺序一致）
    def extract_number(filename):
        import re
        numbers = re.findall(r'\d+', filename)  # 提取文件名中的所有数字
        return numbers[-1] if numbers else "0"  # 保持字符串格式
    
    image_names.sort(key=extract_number)
    image_count = len(image_names)
    
    # 检查是否有图片
    if image_count == 0:
        raise ValueError(f"图片目录中无有效图片！支持格式：{image_extensions}")
    
    print(f"✅ 成功读取 {image_count} 张图片（前5张：{image_names[:5]}）")

    # 交互式标注
    print("\n" + "=" * 50)
    print(f"3. 交互式标注目标：{Config.SINGLE_TARGET[1]}")
    print("=" * 50)
    
    # 读取第1张标注图
    annotate_img_path = os.path.join(Config.IMAGE_SEQ_DIR, image_names[Config.ANNOTATE_IMG_IDX])
    annotate_img = Image.open(annotate_img_path).convert("RGB")
    
    # 开始交互式标注
    annotator = InteractiveAnnotator(annotate_img)
    coords, labels = annotator.start_annotation()
    
    if coords is None or len(coords) == 0:
        print("未进行标注，程序退出")
        return
    
    # 预览标注效果
    plt.figure(figsize=(12, 8))
    plt.title(f"标注结果预览（{Config.SINGLE_TARGET[1]}）", fontsize=12)
    plt.imshow(annotate_img)
    show_points(coords, labels, plt.gca())
    plt.show()
    
    # 确认标注
    confirm = input(f" 确认标注是否正确（y=继续，n=重新标注）：")
    if confirm.lower() != "y":
        print("ℹ请重新运行程序进行标注")
        return

    # 分割所有图片
    print(f"\n4. 分割 {image_count} 张图片（目标：{Config.SINGLE_TARGET[1]}）")
    print("=" * 50)
    
    # 初始化SAM2推理状态（传入图片目录）
    inference_state = predictor.init_state(video_path=Config.IMAGE_SEQ_DIR)
    predictor.reset_state(inference_state)  # 重置缓存，避免干扰
    
    # 添加单目标标注提示
    obj_id = 1
    _, _, _ = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=Config.ANNOTATE_IMG_IDX,  # 在第1张图添加标注
        obj_id=obj_id,
        points=coords,
        labels=labels
    )
    print(f"标注提示添加完成，开始批量分割...")
    
    # 存储所有图片的分割结果
    image_segments = {}
    for img_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
        # 单目标：取第0个掩码（仅1个目标）
        mask = (out_mask_logits[0] > 0.7).cpu().numpy()
        image_segments[img_idx] = mask
        
        # 进度提示（每20张更新一次）
        if (img_idx + 1) % 40 == 0 or img_idx == image_count - 1:
            progress = ((img_idx + 1) / image_count) * 100
            print(f"进度：{img_idx + 1}/{image_count} 张图（{progress:.1f}%）")
    
    print(f"所有 {image_count} 张图分割完成！")

    # 预览分割结果
    print(f"\n5. 预览分割结果（每{Config.VIS_STRIDE}张图显示1张）")
    
    plt.close("all")  # 关闭之前的图，避免内存占用
    for img_idx in range(0, image_count, Config.VIS_STRIDE):
        if img_idx not in image_segments:
            continue
        
        # 读取原图
        img_path = os.path.join(Config.IMAGE_SEQ_DIR, image_names[img_idx])
        img = Image.open(img_path).convert("RGB")
        
        # 获取掩码
        mask = image_segments[img_idx]
        
        # 显示结果
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原图
        axes[0].imshow(img)
        axes[0].set_title(f"原图 #{img_idx+1}: {image_names[img_idx]}")
        axes[0].axis("off")
        
        # 分割结果
        axes[1].imshow(img)
        show_mask(mask, axes[1])
        axes[1].set_title(f"分割结果 #{img_idx+1}")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()

    # 保存结果
    print(f"\n6. 保存分割结果")
    print("=" * 50)
    
    # 创建保存目录
    os.makedirs(Config.SEG_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.CROP_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.masked_save_dir, exist_ok=True)
    os.makedirs(Config.pure_mask_dir, exist_ok=True)
    
    saved_count = 0
    for img_idx in range(image_count):
        if img_idx not in image_segments:
            continue
            
        # 读取原图
        img_path = os.path.join(Config.IMAGE_SEQ_DIR, image_names[img_idx])
        img = Image.open(img_path).convert("RGB")
        mask = image_segments[img_idx]
        
        # 保存带掩码的分割图
        masked_img = add_mask_to_image(img, mask, Config.SINGLE_TARGET[1])
        seg_save_path = os.path.join(Config.SEG_SAVE_DIR, image_names[img_idx])
        masked_img.save(seg_save_path)
        
        # 保存裁剪的目标对象
        cropped_img = crop_object_from_image(img, mask, Config.SINGLE_TARGET[1])
        if cropped_img:
            # RGBA图像必须保存为PNG格式
            crop_filename = image_names[img_idx].replace('.jpg', '.png').replace('.jpeg', '.png').replace('.JPG', '.png').replace('.JPEG', '.png')
            crop_save_path = os.path.join(Config.CROP_SAVE_DIR, crop_filename)
            cropped_img.save(crop_save_path)
        
        # 保存带透明背景的掩码图 
        # masked_transparent = crop_masked_from_image(img, mask, Config.SINGLE_TARGET[1])
        # masked_filename = image_names[img_idx].replace('.jpg', '.png').replace('.jpeg', '.png').replace('.JPG', '.png').replace('.JPEG', '.png')
        # masked_save_path = os.path.join(Config.masked_save_dir, masked_filename)
        # masked_transparent.save(masked_save_path)
        
        # 保存纯掩码（黑白图）
        # 确保掩码是2D数组
        if mask.ndim > 2:
            mask_2d = mask.squeeze()  # 移除多余的维度
        else:
            mask_2d = mask
        
        # 转换为0-255范围的uint8
        pure_mask_array = (mask_2d * 255).astype(np.uint8)
        pure_mask = Image.fromarray(pure_mask_array)
        pure_mask_filename = image_names[img_idx].replace('.jpg', '.png').replace('.jpeg', '.png').replace('.JPG', '.png').replace('.JPEG', '.png')
        pure_mask_path = os.path.join(Config.pure_mask_dir, pure_mask_filename)
        pure_mask.save(pure_mask_path)
        
        saved_count += 1
    
    print(f"保存完成！共处理 {saved_count} 张图片")
    print(f"   - 分割结果图: {Config.SEG_SAVE_DIR}")
    print(f"   - 裁剪对象图: {Config.CROP_SAVE_DIR}")
    print(f"   - 透明背景图: {Config.masked_save_dir}")
    print(f"   - 纯掩码图: {Config.pure_mask_dir}")

if __name__ == "__main__":
    try:
        interactive_segmentation()
    except Exception as e:
        print(f"\n程序出错：{str(e)}")