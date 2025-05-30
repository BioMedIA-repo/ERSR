import torch
import torch.nn.functional as F
from kornia.contrib import distance_transform

from utils import ramps


class GlobalRegularitySelector:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.score_dict = {}  # {global_idx: regularity_score}
        self.topK_ratio = 0

    def update_topk_ratio(self, iter_num):
        ramp = ramps.sigmoid_rampup(iter_num, 10000)
        self.topK_ratio = 0.5 + 0.5 * ramp
        if iter_num >= 10000:
            self.topK_ratio = 1

    def compute_score(self, preds):
        edge_score = compute_edge_smoothness(preds)  # [B]
        edt_score = compute_laplacian_edt_smoothness(preds)  # [B]
        return self.alpha * edge_score + (1 - self.alpha) * edt_score  # [B]

    def update_scores(self, preds, global_indices):
        """
        只更新 score_dict
        """
        scores = self.compute_score(preds).detach().cpu()  # [B]
        for idx, score in zip(global_indices.cpu(), scores):
            self.score_dict[int(idx)] = float(score)

    def select_topk_indices(self, topk_ratio=None):
        if topk_ratio is None:
            topk_ratio = self.topK_ratio
        sorted_items = sorted(self.score_dict.items(), key=lambda x: x[1])  # 小的好
        topk = int(len(sorted_items) * topk_ratio)
        return [item[0] for item in sorted_items[:topk]]

    def filter_batch_by_global_indices(self, imgs, preds, global_indices, selected_global_indices, topK_ratio=0.8):
        """
        只用 batch 内筛选，不需要 buffer。
        """
        device = imgs.device
        batch_size = imgs.size(0)
        selected_set = set(selected_global_indices)

        selected_mask = torch.isin(global_indices, torch.tensor(list(selected_set), device=device))

        selected_imgs = imgs[selected_mask]
        selected_preds = preds[selected_mask]

        expected_num = max(1, int(batch_size * topK_ratio))

        if selected_imgs.size(0) >= expected_num:
            return selected_imgs, selected_preds
        else:
            num_needed = expected_num - selected_imgs.size(0)

            remaining_mask = ~selected_mask
            remaining_imgs = imgs[remaining_mask]
            remaining_preds = preds[remaining_mask]

            if remaining_imgs.size(0) > 0:
                remaining_scores = self.compute_score(remaining_preds).detach().cpu()
                topk_indices = remaining_scores.argsort()[:num_needed]

                supplement_imgs = remaining_imgs[topk_indices]
                supplement_preds = remaining_preds[topk_indices]

                filtered_imgs = torch.cat([selected_imgs, supplement_imgs], dim=0)
                filtered_preds = torch.cat([selected_preds, supplement_preds], dim=0)
            else:
                filtered_imgs = selected_imgs
                filtered_preds = selected_preds

            return filtered_imgs, filtered_preds


def compute_edge_smoothness(preds):
    """
    输入:
        preds: [B, 1, H, W] 单通道预测图（已sigmoid或softmax）
    输出:
        edge_score: [B] 每张图的边缘平滑得分
    """
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=preds.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=preds.device).view(1, 1, 3, 3) / 8.0

    grad_x = F.conv2d(preds, sobel_x, padding=1)  # [B, 1, H, W]
    grad_y = F.conv2d(preds, sobel_y, padding=1)  # [B, 1, H, W]

    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # [B, 1, H, W]
    smooth_score = grad_mag.mean(dim=[1, 2, 3])  # 越平滑得分越低

    return smooth_score  # [B]


def compute_laplacian_edt_smoothness(preds, laplacian_kernel=None):
    """
    对每个预测图像，基于距离变换的 Laplacian 平滑度，值越小越规整。
    返回 [B] tensor
    """
    soft_mask = (preds > 0.5).float()  # binary for distance
    edt_tensor = distance_transform(soft_mask)  # [B, 1, H, W]

    if laplacian_kernel is None:
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32, device=preds.device).view(1, 1, 3, 3)
    else:
        kernel = laplacian_kernel.to(preds.device)

    lap = F.conv2d(edt_tensor, kernel, padding=1)  # [B, 1, H, W]
    return torch.mean(torch.abs(lap), dim=(1, 2, 3))  # shape: [B]

def select_topk_regular(imgs, preds, topk_ratio=0.8, alpha=0.5):
    """
    preds: [B, 1, H, W], labels: [B, 1, H, W]
    返回规整度 TopK 的预测及标签
    alpha: 权重控制 EDT 与边缘平滑度的融合
    """
    B = preds.shape[0]

    edge_score = compute_edge_smoothness(preds)  # [B]
    edt_score = compute_laplacian_edt_smoothness(preds)  # [B]

    # 规整度分数（值越小越规整）
    regular_score = alpha * edge_score + (1 - alpha) * edt_score

    # 选取前 topk% 的索引（越小越规整）
    topk = int(B * topk_ratio)
    _, indices = torch.topk(-regular_score, topk)  # -score => 从小到大排序

    return imgs[indices], preds[indices]
