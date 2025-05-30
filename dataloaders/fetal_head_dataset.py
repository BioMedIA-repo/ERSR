import ast
import itertools

import cv2
import json
import math
import monai
import nibabel as nib
import numpy as np
import os
import random
import re
import time
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from monai import transforms
from monai.data import MetaTensor
from monai.transforms import Compose, RandAffined, RandSpatialCropSamplesd, \
    RandAdjustContrastd, RandGaussianSmoothd, RandHistogramShiftd, \
    RandFlipd, RandScaleIntensityd, RandRotate90d, NormalizeIntensityd, ScaleIntensityd, \
    ScaleIntensityRangePercentilesd, RandGridPatchd, RandCoarseDropoutd, RandGaussianNoised, RandZoomd, Lambda, Lambdad, \
    ToTensor, ToTensord, ScaleIntensityRanged, Zoomd, Resized
from monai.utils import convert_to_tensor
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import exposure
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader


def split_train_test_data(base_dir, image_dir, train_ratio=0.8, random_seed=42):
    """
    从包含数据文件名的文本文件中读取文件名，并根据指定比例划分训练集和测试集。

    :param input_txt_file: 包含文件名的文本文件路径
    :param train_ratio: 训练集占总数据集的比例
    :param random_seed: 随机种子，用于确保结果可重复
    :return: 训练集和测试集的文件名列表
    """
    input_txt_file = os.path.join(base_dir, image_dir, 'path.txt')
    # 读取文件名
    with open(input_txt_file, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    if train_ratio >= 1.0:
        return file_names, None
    else:
        # 使用 train_test_split 划分训练集和测试集
        train_files, test_files = train_test_split(
            file_names,
            train_size=train_ratio,
            random_state=random_seed
        )

        return train_files, test_files


def get_file_list(base_dir, image_dir, mode):
    """
    从包含数据文件名的文本文件中读取文件名，并根据指定比例划分训练集和测试集。

    :param input_txt_file: 包含文件名的文本文件路径
    :param train_ratio: 训练集占总数据集的比例
    :param random_seed: 随机种子，用于确保结果可重复
    :return: 训练集和测试集的文件名列表
    """
    input_txt_file = os.path.join(base_dir, image_dir, f'{mode}.txt')
    # 读取文件名
    with open(input_txt_file, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
        return file_names


class Getfile(Dataset):
    def __init__(self, base_dir, image_dir=None, num_data=None, aug=True, image_list=None, labeled_ratio=None,
                 if_get_labeled=None, seed=None, classes=None):
        self._base_dir = base_dir
        self.image_dir = image_dir
        self.image_list = image_list

        self.aug = aug
        self.num_data = num_data
        self.seed = seed
        self.labeled_ratio = labeled_ratio
        self.classes = classes
        if self.labeled_ratio != 1.0:
            subset_size = int(self.labeled_ratio * len(self.image_list))  # 计算 10% 的数量
            self.labeled_img_list = self.image_list[:subset_size]
            self.unlabeled_img_list = self.image_list[subset_size:]
            # self.labeled_img_list, self.unlabeled_img_list = self.split_labeled_unlabeled_data(self.image_list,
            #                                                                                    labeled_ratio=self.labeled_ratio,
            #                                                                                    random_seed=self.seed)
            total_len = len(self.image_list)
            self.labeled_indices = list(range(0, len(self.labeled_img_list)))
            self.unlabeled_indices = list(range(len(self.labeled_img_list), total_len))

            self.labeled_img_list = [os.path.join(self._base_dir, self.image_dir, file_name) for file_name in
                                     self.labeled_img_list]
            self.unlabeled_img_list = [os.path.join(self._base_dir, self.image_dir, file_name) for file_name in
                                       self.unlabeled_img_list]
            print(f'Loading data from {self.image_dir}, {len(self.labeled_img_list)}, {len(self.unlabeled_img_list)}')
        else:
            print(f'Loading data from {self.image_dir}, {len(self.image_list)}')
            self.labeled_img_list = [os.path.join(self._base_dir, self.image_dir, file_name) for file_name in
                                     self.image_list]

        # self.transform = Compose([
        #     RandZoomd(min_zoom=0.8, max_zoom=1, prob=0.5, keep_size=True, keys=['image', 'label'],
        #               padding_mode='constant', allow_missing_keys=True),
        #     RandFlipd(keys=['image', 'label'], prob=0.5, allow_missing_keys=True),
        #     # RandGaussianSmoothd(prob=0.3, keys=['image'], allow_missing_keys=True),
        #     # RandGaussianNoised(prob=0.3, keys=['image'], allow_missing_keys=True),
        #     # RandAdjustContrastd(gamma=(0.8, 3), prob=0.3, keys=['image'], allow_missing_keys=True),  # 1 0.3
        # ])

        self.transform = Compose([
            RandomGenerator(output_size=(224, 224)),
        ])

        self.resize = Compose([
            Resized(keys=['image', 'label'], spatial_size=(224, 224), mode='nearest'),
        ])

        self.normalize = Compose([
            # Resized(keys=['image', 'label'], spatial_size=(224, 224), mode='nearest'),
            ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0, b_max=1)
        ])

    def split_labeled_unlabeled_data(self, image_list, labeled_ratio=None, random_seed=None):
        """
        从包含数据文件名的文本文件中读取文件名，并根据指定比例划分训练集和测试集。

        :param input_txt_file: 包含文件名的文本文件路径
        :param train_ratio: 训练集占总数据集的比例
        :param random_seed: 随机种子，用于确保结果可重复
        :return: 训练集和测试集的文件名列表
        """
        if self.labeled_ratio >= 1.0:
            return image_list, None
        else:
            # 使用 train_test_split 划分训练集和测试集
            labeled_files, unlabeled_files = train_test_split(
                image_list,
                train_size=self.labeled_ratio,
                random_state=random_seed
            )

            return labeled_files, unlabeled_files

    def _load_npz(self, image_path, label_path):
        data = np.load(image_path)
        data_vol = torch.from_numpy(data['image.npy'].astype(np.float32)).unsqueeze(0).float()
        if label_path is not None:
            label_vol = torch.from_numpy(data['label.npy'].astype(np.float32)).unsqueeze(0).float()
            return data_vol, label_vol
        else:
            return data_vol

    def __len__(self):
        if self.num_data == 0 or self.num_data is None:
            return len(self.image_list)
        return self.num_data

    def __getitem__(self, idx):
        if self.num_data is None or self.num_data == 0:
            idx = idx % len(self.image_list)
        elif idx >= len(self.image_list):
            idx = random.randint(0, len(self.image_list) - 1)

        if idx < len(self.labeled_img_list):
            image_list = self.labeled_img_list
            actual_idx = idx
        else:
            image_list = self.unlabeled_img_list
            actual_idx = idx - len(self.labeled_img_list)

        # 加载图像和标签
        image_path = image_list[actual_idx]

        image, label = self._load_npz(image_path, image_path)
        sample = {'image': image, 'label': label}
        if self.aug:
            sample = self.transform(sample)
        else:
            sample = self.resize(sample)
        sample = self.normalize(sample)
        if self.classes != 1:
            sample['label'] = F.one_hot(sample['label'].long().squeeze(0), 3).permute(2, 0, 1).float()
        # sample['label'] = F.one_hot(sample['label'].long().squeeze(0), 2).permute(2, 0, 1)
        else:
            sample['label'] = (sample['label'] > 0).float()
        # visualize_sample(sample)
        # print(sample['label'].shape, sample['image'].shape)
        # sample = {'image': sample['image'].as_tensor(), 'label': sample['label'].as_tensor()}
        return sample

    def get_filename(self, idx):
        return self.image_list[idx]

    def get_labeled_indices(self):
        """
        返回有标签数据的索引。
        """
        return self.labeled_indices

    def get_unlabeled_indices(self):
        """
        返回无标签数据的索引。
        """
        return self.unlabeled_indices


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        image = image.squeeze(0)
        label = label.squeeze(0)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape

        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8)).unsqueeze(0)
        sample = {"image": image, "label": label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    在每个批次中，分别从主索引流和次索引流中采样指定数量的数据，组合成一个批次。
    主索引流经过一次完整的迭代后（即一个 epoch），会结束迭代。
    次索引流会循环迭代，确保每个批次都能从次索引流中采样到数据。
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices  # 主索引流的索引列表（如有标签数据的索引）。
        self.secondary_indices = secondary_indices  # 次索引流的索引列表（如无标签数据的索引）。
        self.secondary_batch_size = secondary_batch_size  # 次索引流的batch size。
        self.primary_batch_size = batch_size - secondary_batch_size  # 主索引流的batch size。

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)  # 直接打乱主迭代器
        secondary_iter = iterate_eternally(self.secondary_indices)  # 设置无限次迭代器
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
        )
        )

    def __len__(self):  # 用主索引流可以计算出实际批次数量
        return len(self.primary_indices) // self.primary_batch_size


# 打乱并返回新的迭代器
def iterate_once(iterable):
    return np.random.permutation(iterable)


# 无限循环返回随机迭代器
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


# 把迭代器分成batch
def grouper(iterable, batch_size):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * batch_size
    return zip(*args)


def visualize_sample(sample):
    image = sample['image'][0].numpy()  # 调整张量尺寸并转换为NumPy数组
    label = sample['label'][2].numpy()
    # print(label.shape)

    # 可视化图像和标签
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='jet')  # 使用热图可视化标签
    plt.title('Label')
    plt.axis('off')

    plt.show()


def save_image(image, save_dir, path):
    """
    保存图像到指定目录，使用路径的最后一个文件名作为文件名。

    参数:
    image (numpy.ndarray): 要保存的图像。
    save_dir (str): 保存目录。
    path (str): 图像路径，用于获取文件名。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取文件名
    filename = os.path.basename(path)

    # 构建保存路径
    save_path = os.path.join(save_dir, filename + '.png')

    plt.imshow(image.cpu().numpy(), cmap='grey')  # 使用 'jet' colormap
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存 edge_mask 图像
