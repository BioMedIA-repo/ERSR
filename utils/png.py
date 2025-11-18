import os
import cv2
import numpy as np
import re
from glob import glob


def extract_edges(image):
    """提取图像的边缘"""
    edges = cv2.Canny(image, 0.5, 10)
    return edges


def overlay_edges_on_image(base_image, green_edges, red_edges):
    """将绿色边缘和红色边缘叠加到原始图像上"""
    overlay = base_image.copy()
    overlay[green_edges > 0] = [0, 255, 0]  # Green for GT
    overlay[red_edges > 0] = [0, 0, 255]  # Red for Pred
    return overlay


def process_folder(folder_path):
    """处理单个文件夹"""
    all_files = glob(os.path.join(folder_path, "*.png"))

    # 按序号 n 分组
    id_to_files = {}
    for file in all_files:
        filename = os.path.basename(file)
        match = re.match(r"(\d+)_", filename)
        if not match:
            continue
        idx = match.group(1)
        id_to_files.setdefault(idx, []).append(file)
    for idx, files in id_to_files.items():
        img_file = next((f for f in files if "_img" in f), None)
        gt_file = next((f for f in files if "_gt" in f), None)
        pred_files = [f for f in files if os.path.basename(f).startswith(f"{idx}_pred")]

        if not img_file or not gt_file or not pred_files:
            continue

        # 读取图像
        img = cv2.imread(img_file)
        gt = cv2.imread(gt_file)
        green_edges = extract_edges(gt)  # 提取 GT 的边缘

        for pred_file in pred_files:
            pred_img = cv2.imread(pred_file)
            red_edges = extract_edges(pred_img)  # 提取 Pred 的边缘
            overlay_img = overlay_edges_on_image(img, green_edges, red_edges)  # 在 img 上叠加边缘

            # 保存叠加图，直接覆盖
            cv2.imwrite(pred_file, overlay_img)


def traverse_root_folder(root_dir):
    """遍历根目录，处理所有子文件夹"""
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            process_folder(os.path.join(root, d))


if __name__ == "__main__":
    root_directory = "D:\\Desktop\\fetal_head\\se_ab"  # 替换为你的路径
    traverse_root_folder(root_directory)
