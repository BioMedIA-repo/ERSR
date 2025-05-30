import logging
import os
import random
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from dataloaders.fetal_head_dataset import get_file_list, Getfile
from networks.net_factory import net_factory
from utils.get_argparser import get_args
from val import evaluate


def test(args, snapshot_path, seed):
    num_classes = args.num_classes

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    test_files = get_file_list(base_dir=args.root_path, image_dir=args.data_name, mode='test')
    test_dataset = Getfile(base_dir=args.root_path, image_dir=args.data_name, num_data=0, aug=False,
                           image_list=test_files, labeled_ratio=1, if_get_labeled=False, seed=seed, classes=num_classes)
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
    testloader = DataLoader(test_dataset, num_workers=4, shuffle=False, batch_size=16, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    best_model_path = os.path.join(snapshot_path, f'unet_best_model.pth')
    model = create_model(ema=False)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_mean_dice, test_mean_hd95 = evaluate(testloader, model, num_classes)
    return test_mean_dice, test_mean_hd95


def multi_test(args, snapshot_path, num_repeats=3):
    # 用于存储多次训练的结果
    dice_scores = []
    hd95_scores = []

    for i in range(num_repeats):
        print(f"\nStarting training run {i + 1}/{num_repeats}...")
        seed = (i + 1) * 42
        result = test(args, snapshot_path, seed)

        mean_dice, mean_hd95 = result
        dice_scores.append(mean_dice)
        hd95_scores.append(mean_hd95)

    # 转换为 NumPy 数组
    dice_scores = np.array(dice_scores)
    hd95_scores = np.array(hd95_scores)

    # 计算均值和标准差
    dice_mean = np.mean(dice_scores)
    dice_std = np.std(dice_scores)

    hd95_mean = np.mean(hd95_scores)
    hd95_std = np.std(hd95_scores)

    print(f"\nSummary of {args.labeled_ratio} runs:")
    print(f"Dice: {dice_mean:.4f}  {dice_std:.4f}")
    print(f"HD95: {hd95_mean:.4f}  {hd95_std:.4f}")
    return dice_mean, dice_std, hd95_mean, hd95_std


if __name__ == "__main__":
    args = get_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_bs, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    log_file = 'training_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    # 将FileHandler添加到日志记录器
    logger = logging.getLogger('')
    logger.addHandler(file_handler)
    logging.info(str(args))
    multi_test(args, snapshot_path, num_repeats=3)
