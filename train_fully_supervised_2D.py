import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from monai.losses import DiceLoss, DiceFocalLoss
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.fetal_head_dataset import get_file_list, Getfile, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils.get_argparser import get_args
from val2D import evaluate


def train(args, snapshot_path, seed):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    labeled_bs = args.labeled_bs
    labeled_ratio = args.labeled_ratio

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    train_files = get_file_list(base_dir=args.root_path, image_dir=args.data_name, mode='train')
    val_files = get_file_list(base_dir=args.root_path, image_dir=args.data_name, mode='val')
    test_files = get_file_list(base_dir=args.root_path, image_dir=args.data_name, mode='test')

    train_dataset = Getfile(base_dir=args.root_path, image_dir=args.data_name, num_data=0, aug=True,
                            image_list=train_files, labeled_ratio=labeled_ratio, if_get_labeled=True, seed=seed,
                            classes=num_classes)
    val_dataset = Getfile(base_dir=args.root_path, image_dir=args.data_name, num_data=0, aug=False,
                          image_list=val_files, labeled_ratio=1, if_get_labeled=False, seed=seed, classes=num_classes)
    test_dataset = Getfile(base_dir=args.root_path, image_dir=args.data_name, num_data=0, aug=False,
                           image_list=test_files, labeled_ratio=1, if_get_labeled=False, seed=seed, classes=num_classes)

    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)

    if args.labeled_ratio != 1:
        labeled_indices = train_dataset.get_labeled_indices()
        unlabeled_indices = train_dataset.get_unlabeled_indices()
        print(len(labeled_indices), len(unlabeled_indices))
        sampler = TwoStreamBatchSampler(
            primary_indices=labeled_indices,
            secondary_indices=unlabeled_indices,
            batch_size=batch_size,
            secondary_batch_size=batch_size - args.labeled_bs
        )
        trainloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn)
    else:
        trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_dataset, num_workers=4, shuffle=False, batch_size=16, pin_memory=True,
                           worker_init_fn=worker_init_fn)
    testloader = DataLoader(test_dataset, num_workers=4, shuffle=False, batch_size=16, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
    dice_loss = DiceFocalLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    best_model_path = None
    best_model = None
    max_epoch = args.max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            img, label = sampled_batch['image'].cuda(), sampled_batch['label'][:labeled_bs].cuda()
            # 预热阶段仅监督
            seg_out = model(img[:labeled_bs])
            if num_classes == 1:
                seg_out_soft = torch.sigmoid(seg_out)
            else:
                seg_out_soft = torch.softmax(seg_out, dim=1)
            loss_ce = ce_loss(seg_out, label)
            loss_dice = dice_loss(seg_out_soft, label)
            loss = (loss_dice + loss_ce) / 2

            iter_num += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_num > 0 and iter_num % 60 == 0:
                model.eval()
                mean_dice, mean_hd95,_ = evaluate(valloader, model, num_classes,args)

                writer.add_scalar('info/val_mean_dice', mean_dice, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                if mean_dice > best_performance:
                    print('best iter:', iter_num)
                    best_performance = mean_dice
                    best_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), best_model_path)
                model_path = os.path.join(snapshot_path, '{}_model.pth'.format(mean_dice))
                torch.save(model.state_dict(), model_path)
                model.train()
            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            iterator.close()
            break
    writer.close()

    # logging.info(f"Loading best model weights from {best_model_path} for testing...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_mean_dice, test_mean_hd95 = evaluate(testloader, model, num_classes)
    # logging.info(f"Test Results - Mean Dice: {test_mean_dice}, Mean HD95: {test_mean_hd95}")
    return test_mean_dice, test_mean_hd95


def multi_train(args, snapshot_path, num_repeats=1):
    # 用于存储多次训练的结果
    dice_scores = []
    hd95_scores = []

    for i in range(num_repeats):
        print(f"\nStarting training run {i + 1}/{num_repeats}...")
        # 运行训练函数
        # seed = i * 42
        # seed = (i + 1) * 42
        seed = 126
        result = train(args, snapshot_path, seed)

        # 每次训练结束后，在验证集上获取最后的性能指标
        # 这里假设 `train()` 返回最后的验证性能（mean_dice, mean_hd95）
        mean_dice, mean_hd95 = result  # 假设 train() 返回这两个值
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

    # 统一输出所有训练结果
    print("\nResults of all training runs:")
    for i in range(num_repeats):
        print(f"Training {i + 1}: Dice = {dice_scores[i]:.4f}, HD95 = {hd95_scores[i]:.4f}")

    print(f"\nSummary of {args.labeled_ratio} runs:")
    print(f"Dice: {dice_mean:.4f}, Dice: {dice_std:.4f}")
    print(f"HD95: {hd95_mean:.4f}, HD95: {hd95_std:.4f}")
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

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # train(args, snapshot_path)
    multi_train(args, snapshot_path, num_repeats=1)
