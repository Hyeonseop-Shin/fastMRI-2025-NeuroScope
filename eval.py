
import argparse
import os
import sys
from pathlib import Path

from utils.EvalMRI import EvalMRI
from utils.common.utils import seed_fix

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate MRI models on FastMRI challenge',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')
    parser.add_argument("--target_key", type=str, default='image_label', help='Name of target key')
    parser.add_argument("--max_key", type=str, default='max', help='Name of max key')
    parser.add_argument('-key', '--output_key', type=str, default='reconstruction')

    # Data paths
    parser.add_argument('--leaderboard_path', type=Path, default='/root/Data/leaderboard/', help='Path to leaderboard data')
    parser.add_argument('--forward_dir', type=Path, default='/root/fastMRI-2025-NeuroScope/reconstructions_leaderboard', help='Directory for saving reconstructions')
    
    # Model checkpoints
    parser.add_argument('--model', type=str, default='fivarnet', help='Model type to evaluate')
    parser.add_argument('--brain_acc4_ckpt', type=Path, default='/root/fastMRI-2025-NeuroScope/results/fivarnet_f8_i2_c32_s8_epoch5_fold5_seed2025_acc4-brain/checkpoints/best_model.pt', help='Checkpoint for brain acc4 model')
    parser.add_argument('--brain_acc8_ckpt', type=Path, default='../result/eval_mri/checkpoints/brain_acc8.pth', help='Checkpoint for brain acc8 model')
    parser.add_argument('--knee_acc4_ckpt', type=Path, default='../result/eval_mri/checkpoints/knee_acc4.pth', help='Checkpoint for knee acc4 model')
    parser.add_argument('--knee_acc8_ckpt', type=Path, default='../result/eval_mri/checkpoints/knee_acc8.pth', help='Checkpoint for knee acc8 model')

    # model hyperparameter
    parser.add_argument('-f', '--feature_cascades', type=int, default=8, help='Number of cascades | Should be less than 12')
    parser.add_argument('-i', '--image_cascades', type=int, default=2, help='Number of cascades | Should be less than 12')
    parser.add_argument('-a', '--use_attention', type=str2bool, default=False, choices=[True, False], help='Applying block-wise attention for feature processor')
    parser.add_argument('--chans', type=int, default=32, help='Number of channels for cascade U-Net | 18 in original varnet')
    parser.add_argument('--sens-chans', type=int, default=8, help='Number of channels for sensitivity map U-Net | 8 in original varnet')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()


    task = EvalMRI(args)
    # task.shit(data_path=Path("/root/Data/leaderboard/acc4/"), recon_anatomy='brain')
    task.partial_reconstruction(acc=4, anatomy='brain')
    task.partial_lb_eval(acc=4, anatomy='brain')