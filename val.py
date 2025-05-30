import logging
import numpy as np
import torch
from medpy import metric

from utils import binary


def evaluate(val_dataset, net, num_classes):
    net.eval()
    all_batch_dice = []
    all_batch_jc = []
    all_batch_hd = []
    all_batch_asd = []
    logger = logging.getLogger('')
    for idx, batch in enumerate(val_dataset):
        xt, xt_labels = batch['image'].cuda(), batch['label'].cuda()
        with torch.no_grad():
            out = net(xt)
            if num_classes == 1:
                out = torch.sigmoid(out).squeeze(0)
                out = (out > 0.5).to(dtype=torch.float32)
        for ind in range(out.shape[0]):
            out_img = out[ind]
            xt_lab_img = xt_labels[ind]
            if torch.sum(xt_lab_img) == 0:
                continue

            pred = (out_img == 1).cpu().numpy()
            gt = (xt_lab_img == 1).cpu().numpy()

            dice, jc, hd, asd = calculate_metric_percase(pred, gt)
            all_batch_dice.append(dice)
            all_batch_jc.append(jc)
            all_batch_hd.append(hd)
            all_batch_asd.append(asd)

        # 转换为 numpy 数组
    all_batch_dice = np.array(all_batch_dice)
    all_batch_jc = np.array(all_batch_jc)
    all_batch_hd = np.array(all_batch_hd)
    all_batch_asd = np.array(all_batch_asd)

    total_mean_dice = np.mean(all_batch_dice)
    total_mean_jc = np.mean(all_batch_jc)
    total_mean_hd = np.mean(all_batch_hd)
    total_mean_asd = np.mean(all_batch_asd)
    print(
        f'Dc: {total_mean_dice:.3f}, Jc: {total_mean_jc:.5f}, HD: {total_mean_hd:.5f}, ASD: {total_mean_asd:.5f}')

    return total_mean_dice, total_mean_jc, total_mean_hd, total_mean_asd


def calculate_metric_percase(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            dice = binary.dc(pred, gt) * 100
            jc = binary.jc(pred, gt)
            hd = binary.hd95(pred, gt)
            asd = binary.asd(pred, gt)
            return dice, jc, hd, asd
        except RuntimeError as e:
            logging.error(f"Error calculating metrics: {e}")
            return 0, 0, 0, 0
    else:
        return 0, 0, 0, 0
