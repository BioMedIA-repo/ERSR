import os
import numpy as np
from PIL import Image
import torch


def save_images_with_derivatives(batch_images, save_folder, iter_folder, file_name):
    """
    保存批量图像及其衍生图像到指定文件夹。

    参数：
    - batch_images: numpy.ndarray 或 torch.Tensor，形状为 [B, C, H, W] 的批量图像。
    - save_folder: str，主保存文件夹路径。
    - iter_folder: str，子文件夹名称（通常是 batch 编号）。
    - file_name: str，保存的图像文件名前缀（不含扩展名）。
    """
    # 确保 batch_images 是 numpy 数组
    if isinstance(batch_images, torch.Tensor):
        batch_images = batch_images.clone().detach().cpu().numpy()  # 如果是 torch.Tensor，转换为 numpy

    # 检查图像形状是否符合 [B, C, H, W]
    if len(batch_images.shape) != 4:
        raise ValueError("输入的图像形状必须为 [B, C, H, W]")

    # 创建子文件夹路径
    batch_path = os.path.join(save_folder, str(iter_folder))
    os.makedirs(batch_path, exist_ok=True)

    # 遍历保存每张图像
    for i in range(batch_images.shape[0]):  # 遍历 batch
        # 获取单张图像，形状为 [C, H, W]
        image_array = batch_images[i]

        # 如果是单通道图像（灰度图像）
        if image_array.shape[0] == 1:
            # 去掉通道维度，形状变为 [H, W]
            image_array = image_array[0]
            # 将图像数据归一化到 [0, 255]，并转换为 uint8 类型
            image_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))) * 255
            image_array = np.clip(image_array, 0, 255)
            image_array = image_array.astype(np.uint8)

            # 创建 PIL 图像对象
            image = Image.fromarray(image_array)

        # 如果是三通道图像（彩色图像）
        elif image_array.shape[0] == 3:
            # 转换为 [H, W, C] 形状
            image_array = np.transpose(image_array, (1, 2, 0))
            # 将图像数据归一化到 [0, 255]，并转换为 uint8 类型
            image_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))) * 255
            image_array = np.clip(image_array, 0, 255)
            image_array = image_array.astype(np.uint8)

            # 创建 PIL 图像对象
            image = Image.fromarray(image_array)

        else:
            raise ValueError("输入的图像通道数必须为 1（灰度图像）或 3（彩色图像）")

        # 保存图像
        image_path = os.path.join(batch_path, f"{i}_{file_name}.png")
        image.save(image_path)
        print(f"图像已保存到: {image_path}")
