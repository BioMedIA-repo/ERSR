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
import torch.nn.functional as F
import torch.optim as optim
from monai.losses import DiceLoss, DiceFocalLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataloaders.fetal_head_dataset import get_file_list, Getfile, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import ramps
from utils.get_argparser import get_args
from utils.seletor import GlobalRegularitySelector
from utils.util import split_and_reflect_by_axis, \
    get_mirror_label, apply_mirror_augmentation, \
    fit_ellipse_and_optimize_pseudo_label
from val import evaluate

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    model_state_dict = model.state_dict()
    ema_state_dict = ema_model.state_dict()

    for name, ema_param in ema_state_dict.items():
        if name in model_state_dict:  # 仅更新参数名匹配的项
            param = model_state_dict[name]
            ema_param = ema_param.to(torch.float32)
            param = param.to(torch.float32)
            ema_param.mul_(alpha).add_(1 - alpha, param)


def get_aug_data(img, labels, symmetry_axis):
    left_half, left_mirror, right_half, right_mirror = split_and_reflect_by_axis(labels, symmetry_axis)
    left_mirror_label, right_mirror_label = get_mirror_label(left_half, left_mirror, right_half, right_mirror)
    left_aug, right_aug = apply_mirror_augmentation(img, left_half, left_mirror, right_half, right_mirror,
                                                    symmetry_axis, random_perturbation=True)
    aug_imgs = torch.cat([left_aug, right_aug], dim=0)
    aug_labels = torch.cat([left_mirror_label, right_mirror_label], dim=0)
    return aug_imgs, aug_labels, left_aug, right_aug


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
    ema_model = create_model(ema=True)
    sup_model_path = os.path.join('./pretrained_ckpt', f'{args.pth}_unet_best_model.pth')
    model.load_state_dict(torch.load(sup_model_path))
    ema_model.load_state_dict(torch.load(sup_model_path))

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

    labeled_indices = train_dataset.get_labeled_indices()
    unlabeled_indices = train_dataset.get_unlabeled_indices()
    print(len(labeled_indices), len(unlabeled_indices))

    sampler = TwoStreamBatchSampler(
        primary_indices=labeled_indices,
        secondary_indices=unlabeled_indices,
        batch_size=batch_size,
        secondary_batch_size=batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)

    trainloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_dataset, num_workers=4, shuffle=False, batch_size=16, pin_memory=True,
                           worker_init_fn=worker_init_fn)
    testloader = DataLoader(test_dataset, num_workers=4, shuffle=False, batch_size=16, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    lr_lambda = lambda iter_num: (1.0 - iter_num / args.max_iterations) ** 0.9
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    ce_loss = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
    dice_loss = DiceFocalLoss()
    mse_loss = nn.MSELoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance = 0.0
    best_model_path = None
    best_model = None
    max_epoch = args.max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    consistency_weight = 0
    for epoch in iterator:
        global_selector = GlobalRegularitySelector(alpha=0.5)
        for i_batch, sampled_batch in enumerate(trainloader):
            img, label = sampled_batch['image'].cuda(), sampled_batch['label'][:labeled_bs].cuda()
            unlabeled_img = sampled_batch['image'][labeled_bs:].cuda()
            # 预热阶段仅监督
            seg_out = model(img[:labeled_bs])
            seg_out_soft = torch.sigmoid(seg_out)
            loss_ce = ce_loss(seg_out, label)
            loss_dice = dice_loss(seg_out_soft, label)
            sup_loss = (loss_dice + loss_ce) / 2

            consistency_weight = get_current_consistency_weight(iter_num)
            with torch.no_grad():
                ema_model.eval()
                ema_output = ema_model(unlabeled_img)
                ema_output_soft = torch.sigmoid(ema_output)
                ema_model.train()
                # 获取增强图像
            # 1、形状感知筛选
            # === Global Regularity 筛选 ===
            b = unlabeled_img.size(0)
            global_indices = torch.arange(i_batch * b, (i_batch + 1) * b).to(
                unlabeled_img.device)
            global_selector.update_topk_ratio(iter_num)
            global_selector.update_scores(ema_output_soft, global_indices)
            topk_indices = global_selector.select_topk_indices(topk_ratio=global_selector.topK_ratio)

            unlabeled_img, ema_output_soft = global_selector.filter_batch_by_global_indices(
                unlabeled_img, ema_output_soft, global_indices, topk_indices
            )
            # 2、伪标签优化
            ema_output_soft, _, _, symmetry_axis = fit_ellipse_and_optimize_pseudo_label(ema_output_soft)
            aug_imgs, _, _, _ = get_aug_data(unlabeled_img, ema_output_soft, symmetry_axis)
            all_aug_imgs = torch.cat([unlabeled_img, aug_imgs], dim=0)
            # 学生模型预测
            aug_seg_out = model(all_aug_imgs)
            aug_seg_out_soft = torch.sigmoid(aug_seg_out)
            # 伪标签分割损失
            seg_loss = mse_loss(aug_seg_out_soft[:len(unlabeled_img)], ema_output_soft)
            # 3、基于对称的一致性则正则化,aug_img是left+right
            left_half, left_mirror, right_half, right_mirror = split_and_reflect_by_axis(
                aug_seg_out_soft[:len(unlabeled_img)], symmetry_axis)
            left_mirror_label, right_mirror_label = get_mirror_label(left_half, left_mirror, right_half,
                                                                     right_mirror)
            out_sum = torch.cat([left_mirror_label, right_mirror_label], dim=0)
            aug_loss = mse_loss(aug_seg_out_soft[len(unlabeled_img):], out_sum)
            aug_symmetry_axis_batch = torch.cat([symmetry_axis, symmetry_axis], dim=0)
            out_left_half, _, _, out_right_half_mirror = split_and_reflect_by_axis(
                aug_seg_out_soft[len(unlabeled_img):], aug_symmetry_axis_batch)
            mirror_loss = mse_loss(out_left_half, out_right_half_mirror)
            consistency_loss = seg_loss + aug_loss + mirror_loss
            loss = sup_loss + consistency_weight * consistency_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            writer.add_scalar('info/lr', scheduler.get_last_lr(), iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            consistency_loss = torch.tensor(consistency_loss, dtype=torch.float)
            writer.add_scalar('info/consistency_loss', consistency_weight * consistency_loss.item(), iter_num)
            consistency_weight = torch.tensor(consistency_weight, dtype=torch.float)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            scheduler.step()

            # if iter_num % 200 == 0:
            #     if iter_num < 3000:
            #         image = img[0, 0:1, :, :]
            #         left_aug_img = left_imgs[0, 0:1, :, :]
            #         right_aug_img = right_imgs[0, 0:1, :, :]
            #         concat_image = torch.cat([image, left_aug_img, right_aug_img], dim=2)
            #         writer.add_image('train/Img', concat_image.as_tensor(), iter_num)
            #     if iter_num > 3001:
            #         image = unlabeled_img[0, 0:1, :, :]
            #         left_aug_img = left_imgs[0, 0:1, :, :]
            #         right_aug_img = right_imgs[0, 0:1, :, :]
            #         concat_image = torch.cat([image, left_aug_img, right_aug_img], dim=2)
            #         writer.add_image('train/Img', concat_image.as_tensor(), iter_num)
            #         aug_seg_out_tb = (aug_seg_out_soft > 0.5)
            #         left_half_img = out_left_half[0, ...]
            #         left_mirror_img = out_right_half_mirror[0, ...]
            #         left = torch.cat([left_half_img, left_mirror_img], dim=2)
            #         aug_seg_out_tb_img = aug_seg_out_tb[0, ...]
            #         writer.add_image('train/Left', left.as_tensor(), iter_num)
            #         writer.add_image('train/out', aug_seg_out_tb_img.as_tensor(), iter_num)
            #         # ema_output_soft = (ema_output_soft > 0.5).float()
            #         # writer.add_image('train/ema_out', ema_output_soft[0, ...].as_tensor(), iter_num)
            #         writer.add_image('train/pseudo_labels', labels_refine[0, ...], iter_num)
            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                mean_dice, total_mean_jc,mean_hd95,total_mean_asd = evaluate(valloader, model, num_classes)
                writer.add_scalar('info/val_mean_dice', mean_dice, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                if mean_dice > best_performance:
                    print('best iter:', iter_num)
                    best_performance = mean_dice
                    best_model = model
                    best_iter = iter_num
                    best_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(seed))
                    torch.save(model.state_dict(), best_model_path)
                model.train()
            iter_num += 1
            if iter_num >= args.max_iterations:
                break

        if iter_num >= args.max_iterations:
            iterator.close()
            break

    writer.close()

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print('Best...', best_iter)
    total_mean_dice, total_mean_jc, total_mean_hd, total_mean_asd = evaluate(testloader, model, num_classes)
    return total_mean_dice, total_mean_jc, total_mean_hd, total_mean_asd


def multi_train(args, snapshot_path, num_repeats=3):
    dice_scores = []
    hd95_scores = []

    for i in range(num_repeats):
        print(f"\nStarting training run {i + 1}/{num_repeats}...")
        seed = (i + 1) * 42
        result = train(args, snapshot_path, seed)

        mean_dice, mean_hd95 = result
        dice_scores.append(mean_dice)
        hd95_scores.append(mean_hd95)

    dice_scores = np.array(dice_scores)
    hd95_scores = np.array(hd95_scores)
    dice_mean = np.mean(dice_scores)
    dice_std = np.std(dice_scores)
    hd95_mean = np.mean(hd95_scores)
    hd95_std = np.std(hd95_scores)

    print("\nResults of all training runs:")
    for i in range(num_repeats):
        print(f"Training {i + 1}: Dice = {dice_scores[i]:.4f}, HD95 = {hd95_scores[i]:.4f}")

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
    logger = logging.getLogger('')
    logger.addHandler(file_handler)
    logging.info(str(args))
    # train(args, snapshot_path)
    multi_train(args, snapshot_path, num_repeats=3)
