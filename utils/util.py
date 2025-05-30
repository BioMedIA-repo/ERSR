import os
import pickle
import random

import cv2
import numpy as np
import re

from kornia.contrib import distance_transform
from kornia.filters import sobel
from monai.transforms import (
    RandAdjustContrastd, RandGaussianNoised, RandShiftIntensityd,
    Compose, MapTransform, RandGaussianSmoothd
)
from scipy.ndimage import distance_transform_edt as distance
from scipy.optimize import leastsq
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import torch.nn.functional as F
import networks


def generate_edge_overlay(input_images, labels, thickness=2):
    """
    提取标签边缘，向内扩充指定厚度，并确保扩展后的边缘不超出标签范围。
    将生成的轮廓覆盖到输入图像上。

    Args:
        input_images (torch.Tensor): 输入图像，形状为 (B, C, H, W)。
        labels (torch.Tensor): 标签图像，形状为 (B, 1, H, W)，值为 0 和 1。
        thickness (int): 轮廓的厚度，默认为 10 像素。

    Returns:
        torch.Tensor: 添加轮廓的图像，形状为 (B, C, H, W)。
    """
    device = input_images.device  # 确保在 GPU 上运行
    B, C, H, W = input_images.shape

    # 检查输入的形状
    assert labels.shape == (B, 1, H, W), "标签的形状必须为 (B, 1, H, W)"

    # 计算标签的边缘（通过 Sobel 算子）
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)

    edge_x = F.conv2d(labels.float(), sobel_x, padding=1)
    edge_y = F.conv2d(labels.float(), sobel_y, padding=1)
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)  # 边缘强度

    # 将边缘二值化（将边缘值限制在 0 和 1 之间）
    edges_binary = (edges > 0).float()

    # 使用膨胀操作扩展边缘厚度
    kernel = torch.ones((1, 1, thickness * 2 + 1, thickness * 2 + 1), device=device)
    edges_dilated = F.conv2d(edges_binary, kernel, padding=thickness)
    edges_dilated = (edges_dilated > 0).float()  # 再次二值化

    # 约束扩展后的边缘，使其不超出原始标签范围
    edges_constrained = edges_dilated * labels.float()  # 与标签范围相交

    # 将约束后的边缘覆盖到输入图像上，覆盖值为图像的最大值
    overlay = input_images.clone()
    for b in range(B):
        for c in range(C):
            # 计算当前通道的最大值
            max_value = torch.max(input_images[b, c])
            # 将边缘区域的值替换为最大值
            overlay[b, c] = torch.where(edges_constrained[b, 0] > 0, max_value, overlay[b, c])
    return overlay


def select_largest_connected_component(binary_label):
    """
    选择二值化伪标签的最大连通区域。

    Args:
        binary_label (numpy.ndarray): 二值化后的伪标签图，值为 {0, 1}。

    Returns:
        largest_component (numpy.ndarray): 仅包含最大连通区域的二值图。
    """
    # 寻找连通区域
    num_labels, labels = cv2.connectedComponents(binary_label)

    # 统计每个连通区域的面积
    area_counts = np.bincount(labels.flatten())

    # 排除背景区域（label=0）
    area_counts[0] = 0

    # 找到最大连通区域的标签
    largest_label = np.argmax(area_counts)

    # 生成仅包含最大连通区域的二值图
    largest_component = (labels == largest_label).astype(np.uint8)
    return largest_component

def fit_ellipse_and_optimize_pseudo_label(pseudo_label_batch, threshold=0.6, device='cuda'):
    # 将输入转换为 numpy 数组（如果是 tensor）
    if isinstance(pseudo_label_batch, torch.Tensor):
        pseudo_label_batch = pseudo_label_batch.detach().cpu().numpy()

    B, C, H, W = pseudo_label_batch.shape
    pseudo_label_batch = pseudo_label_batch.squeeze(1)  # 去掉通道维度
    optimized_pseudo_label_batch = np.zeros_like(pseudo_label_batch)  # 初始化优化伪标签# 初始化椭圆区域
    ellipse_angle_list = []
    x_rot_axis_batch = []
    symmetry_axis_list = []

    for i in range(B):  # 逐张处理
        pseudo_label = pseudo_label_batch[i]

        # Step 1: 将概率值从 [0, 1] 转换到 [0, 255]
        pseudo_label_scaled = (pseudo_label * 255).astype(np.uint8)

        # Step 2: 二值化伪标签
        binary_label = (pseudo_label_scaled > (threshold * 255)).astype(np.uint8)
        binary_label = select_largest_connected_component(binary_label)

        # Step 3: 找到伪标签的轮廓
        contours, _ = cv2.findContours(binary_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Step 4: 使用最大轮廓拟合椭圆
        merged_contour = max(contours, key=cv2.contourArea)
        if len(contours) == 0 or len(merged_contour) < 5:
            # 使用默认对称轴（例如图像中心为对称轴）
            x_default_rot = np.tile(np.arange(W) - W // 2, (H, 1)).astype(np.float32)
            x_rot_axis_batch.append(x_default_rot)
            ellipse_angle_list.append(0.0)
            symmetry_axis_list.append([W / 2, H / 2, 1.0, 0.0])
            continue

        ellipse = cv2.fitEllipse(merged_contour)
        (cx, cy), (major_axis, minor_axis), angle = ellipse

        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        angle_rad = np.deg2rad(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 旋转坐标系
        x_rot = (x - cx) * cos_angle + (y - cy) * sin_angle
        y_rot = -(x - cx) * sin_angle + (y - cy) * cos_angle
        x_rot_axis_batch.append(x_rot)  # 保存每张图的 x_rot 坐标
        ellipse_angle_list.append(angle)  # 保存旋转角
        symmetry_axis_list.append([cx, cy, cos_angle, sin_angle])

        # 椭圆距离计算
        ellipse_distance = (x_rot / (major_axis / 2)) ** 2 + (y_rot / (minor_axis / 2)) ** 2
        # Step 6: 优化伪标签
        optimized_pseudo_label = pseudo_label.copy()
        # 对椭圆内部区域进行整体计算（越接近中心点，填充值越高）
        ellipse_region = (ellipse_distance <= 1)  # 椭圆内部所有区域

        # 1. 增加填充基准值
        # 2. 使用非线性提升填充值
        base_fill_value = 0.6  # 基准填充值，确保即使距离较远也有较高值
        nonlinear_factor = 2  # 非线性提升因子（平方或指数）

        # 计算填充值：距离越接近中心，填充值越高
        fill_value = base_fill_value + (1 - ellipse_distance[ellipse_region]) ** nonlinear_factor
        fill_value = np.clip(fill_value, 0, 1)  # 防止填充值超过 1

        # 更新伪标签：取填充值和预测值的较大值
        optimized_pseudo_label[ellipse_region] = np.maximum(
            optimized_pseudo_label[ellipse_region], fill_value)
        # 对超出椭圆边界的区域降低概率值
        outside_region = ellipse_distance > 1
        decay_factor = 0.5  # 控制衰减幅度的因子
        optimized_pseudo_label[outside_region] *= (np.exp(-decay_factor * (ellipse_distance[outside_region] - 1)))

        optimized_pseudo_label_batch[i] = optimized_pseudo_label

        # 转换为 Tensor 并返回
    optimized_tensor = torch.from_numpy((optimized_pseudo_label_batch > threshold).astype(np.float32)).unsqueeze(1).to(
        device)
    x_rot_axis_batch = torch.from_numpy(np.stack(x_rot_axis_batch, axis=0)).to(device)  # (B, H, W)
    ellipse_angle_batch = torch.tensor(ellipse_angle_list, dtype=torch.float32).to(device)  # (B,)
    symmetry_axis_batch = torch.from_numpy(np.stack(symmetry_axis_list, axis=0)).to(device)

    return optimized_tensor, x_rot_axis_batch, ellipse_angle_batch, symmetry_axis_batch


def reflect_by_symmetry_axis(mask_batch: torch.Tensor, symmetry_axis_batch: torch.Tensor):
    """
    关于对称轴 (cx, cy, dx, dy) 进行镜像反射。
    """
    B, _, H, W = mask_batch.shape
    device = mask_batch.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'
    )
    x = x.view(1, H, W).expand(B, -1, -1)
    y = y.view(1, H, W).expand(B, -1, -1)

    cx, cy, dx, dy = symmetry_axis_batch[:, 0], symmetry_axis_batch[:, 1], symmetry_axis_batch[:,
                                                                           2], symmetry_axis_batch[:, 3]

    cx = cx.view(B, 1, 1)
    cy = cy.view(B, 1, 1)
    dx = dx.view(B, 1, 1)
    dy = dy.view(B, 1, 1)

    # 归一化方向向量
    norm = torch.sqrt(dx ** 2 + dy ** 2)
    dx = dx / norm
    dy = dy / norm
    # 如果 (dx, dy) 是法向量，则需要旋转 90° 得到方向向量
    dx, dy = -dy, dx

    # 平移后的坐标向量 v
    vx = x - cx
    vy = y - cy

    # 向量投影 v ⋅ d
    dot = vx * dx + vy * dy

    # 2 * 投影向量 - 原始向量 = 对称向量
    projx = dot * dx
    projy = dot * dy

    reflx = 2 * projx - vx
    refly = 2 * projy - vy

    # 加回中心点坐标
    sym_x = reflx + cx
    sym_y = refly + cy

    # 最近邻采样
    sym_x_clamped = sym_x.round().clamp(0, W - 1).long()
    sym_y_clamped = sym_y.round().clamp(0, H - 1).long()

    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, H, W)
    reflected_mask = mask_batch[batch_indices, 0, sym_y_clamped, sym_x_clamped].unsqueeze(1)

    return reflected_mask


def split_and_reflect_by_axis(mask_batch, symmetry_axis_batch):
    """
    根据旋转椭圆对称轴拆分 mask，得到左半/右半区域的镜像增强版本。
    """
    B, _, H, W = mask_batch.shape
    device = mask_batch.device

    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.view(1, H, W).expand(B, -1, -1)
    y = y.view(1, H, W).expand(B, -1, -1)

    cx, cy, dx, dy = symmetry_axis_batch[:, 0], symmetry_axis_batch[:, 1], symmetry_axis_batch[:,
                                                                           2], symmetry_axis_batch[:, 3]
    cx = cx.view(B, 1, 1)
    cy = cy.view(B, 1, 1)
    dx = dx.view(B, 1, 1)
    dy = dy.view(B, 1, 1)

    px = x - cx
    py = y - cy
    proj = px * dx + py * dy  # x_rot: 投影到对称轴方向

    left_mask = mask_batch * (proj < 0).unsqueeze(1)
    right_mask = mask_batch * (proj >= 0).unsqueeze(1)

    left_reflected = reflect_by_symmetry_axis(left_mask, symmetry_axis_batch)
    right_reflected = reflect_by_symmetry_axis(right_mask, symmetry_axis_batch)

    return left_mask, left_reflected, right_mask, right_reflected


def apply_mirror_augmentation(image, left_half_mask, left_mirror_mask, right_half_mask, right_mirror_mask,
                              symmetry_axis_batch, random_perturbation=True):
    """
    基于对称轴角度与位置，执行对称增强操作，增强图像左右前景区域。
    image: (B, 1, H, W)
    x_rot_axis_batch: (B, H, W)
    angle_batch: (B,) 角度（弧度）
    """
    B, _, H, W = image.shape
    device = image.device
    # 提取左右部分
    left_half = image * left_half_mask
    right_half = image * right_half_mask
    left_mirror = reflect_by_symmetry_axis(left_half, symmetry_axis_batch)
    right_mirror = reflect_by_symmetry_axis(right_half, symmetry_axis_batch)

    # 背景提取（非前景区域）
    foreground_mask = ((left_half_mask + right_half_mask) > 0).float()
    background = image * (1.0 - foreground_mask)
    if random_perturbation:
        # 随机扰动
        left_half = apply_random_pixel_perturbation(left_half, left_half_mask)
        left_mirror = apply_random_pixel_perturbation(left_mirror, left_mirror_mask)
        right_half = apply_random_pixel_perturbation(right_half, right_half_mask)
        right_mirror = apply_random_pixel_perturbation(right_mirror, right_mirror_mask)

    # 最终增强图像
    left_aug = background + left_half + left_mirror
    right_aug = background + right_half + right_mirror

    return left_aug, right_aug, left_half, left_mirror, right_half, right_mirror


def apply_random_pixel_perturbation(image, mask, mode_probs=None):
    """
    使用 MONAI 对图像指定区域进行像素级扰动（噪声、对比度、亮度、模糊），由 MONAI 控制概率。

    Args:
        image: (B, 1, H, W) tensor, 灰度图
        mask:  (B, 1, H, W) tensor, 掩码（值为 0 或 1）
        mode_probs: dict，扰动模式及其概率，如 {'noise': 0.3, 'contrast': 0.3, 'brightness': 0.2, 'blur': 0.3}

    Returns:
        扰动后的图像 tensor
    """
    if mode_probs is None:
        mode_probs = {
            'noise': 0.2,
            'contrast': 0.2,
            'brightness': 0.2,
            'blur': 0.2
        }

    transforms_list = [
        ApplyMaskedTransform(
            keys=["image"], mask_key="mask",
            transform=RandGaussianNoised(keys=["image"], prob=mode_probs.get('noise', 0.0), mean=0.0, std=0.05)
        ),
        ApplyMaskedTransform(
            keys=["image"], mask_key="mask",
            transform=RandAdjustContrastd(keys=["image"], prob=mode_probs.get('contrast', 0.0), gamma=(1.2, 1.8))
        ),
        ApplyMaskedTransform(
            keys=["image"], mask_key="mask",
            transform=RandShiftIntensityd(keys=["image"], prob=mode_probs.get('brightness', 0.0), offsets=0.1)
        ),
        ApplyMaskedTransform(
            keys=["image"], mask_key="mask",
            transform=RandGaussianSmoothd(keys=["image"], prob=mode_probs.get('blur', 0.0), sigma_x=(0.5, 1.5),
                                          sigma_y=(0.5, 1.5))
        ),
    ]

    pipeline = Compose(transforms_list)
    data = {"image": image, "mask": mask}
    perturbed = pipeline(data)["image"]
    return perturbed

def get_mirror_label(left_half, left_mirror, right_half, right_mirror):
    left_label = left_half + left_mirror
    right_label = right_half + right_mirror
    return left_label, right_label


class ApplyMaskedTransform(MapTransform):
    """
    对图像的指定 mask 区域施加变换（支持单通道图像）。
    """

    def __init__(self, keys, mask_key, transform):
        super().__init__(keys)
        self.mask_key = mask_key
        self.transform = transform

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            mask = d[self.mask_key]  # (B, 1, H, W)
            image = d[key]  # (B, 1, H, W)
            B = image.shape[0]
            transformed = []
            for i in range(B):
                img = image[i]  # (1, H, W)
                msk = mask[i]  # (1, H, W)
                perturbed = self.transform({'image': img})['image']
                result = img * (1 - msk) + perturbed * msk
                # 最大最小归一化
                result = (result - result.min()) / (result.max() - result.min() + 1e-8)
                transformed.append(result)
            d[key] = torch.stack(transformed, dim=0)
        return d
