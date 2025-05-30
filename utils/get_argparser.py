import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data2D', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='se/Sup', help='cu/se')
    parser.add_argument('--data_name', type=str,
                        default='fetal_head_se', help='fetal_head_cu, fetal_head_se, Name of Experiment')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--max_iterations', type=int,
                        default=10000, help='maximum epoch number to train')
    parser.add_argument('--pth', type=str, default='cu_01', help='pretrained model')

    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list, default=[224, 224],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_classes', '-cls', type=int, default=1, help='output channel of network')

    # label and unlabel
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_bs per batch')
    parser.add_argument('--labeled_ratio', type=float, default=1, help='labeled data')
    # costs
    parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=10000.0, help='consistency_rampup')

    args = parser.parse_args()
    return args
