
import argparse
import os
import sys
from pathlib import Path
from utils.FastMRI import FastMRI

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(0, os.getcwd() + '/utils/model/')

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(0, os.getcwd() + '/utils/common/')
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


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    
    # Training hyperparameter
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('-e', '--num-epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-n', '--net-name', type=Path, default='fivarnet', help='Name of network')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--criterion', type=str, default='SSIM', choices=['SSIM', 'SSIM_L1'], help='criterion')
    parser.add_argument('--accumulation-step', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--retrain', type=str2bool, default=False, help="retrain from trained model")

    # scheduler hyperparameter
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'constant', 'warmup_cosine', 'double_warmup_cosine'], help='LR scheduler type')
    parser.add_argument('--lr', type=float, default=3e-4, help='Max learning rate (after warmup)')
    parser.add_argument('--lr-min1', type=float, default=0.00005, help='Min LR after first annealing')
    parser.add_argument('--lr-max2', type=float, default=0.00015, help='Max LR for second warmup')
    parser.add_argument('--lr-min2', type=float, default=0.0, help='Min LR after second annealing')
    parser.add_argument('--warmup1', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--anneal1', type=int, default=40, help='First cosine annealing epochs')
    parser.add_argument('--warmup2', type=int, default=10, help='Second warmup epochs')
    parser.add_argument('--anneal2', type=int, default=40, help='Second cosine annealing epochs')
    
    # Data hyperparameter
    parser.add_argument('-d', '--data-augmentation', type=str2bool, default=False, help='Apply spatial augmentation')
    parser.add_argument('-t', '--data-path-train', type=Path, default='D://Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='D://Data/val/', help='Directory of validation data')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    
    # MoE hyperparameter
    parser.add_argument('--use-moe', type=str2bool, default=True, help='Use Mixture of Experts training')
    parser.add_argument('--class-split-path', type=Path, default="D://Data/class_indices", help="Class indicating file location")

    # K-Fold hyperparameter
    parser.add_argument('--k-fold', type=str2bool, default=True, help='Use K-Fold cross-validation')
    parser.add_argument('--num-folds', type=int, default=10, help='Number of folds for K-Fold cross-validation')

    # model hyperparameter
    parser.add_argument('-m', '--model', type=str, default='fivarnet', choices=['varnet', 'fivarnet'], help='Model type')
    parser.add_argument('-f', '--feature_cascades', type=int, default=3, help='Number of cascades | Should be less than 12')
    parser.add_argument('-i', '--image_cascades', type=int, default=3, help='Number of cascades | Should be less than 12')
    parser.add_argument('-a', '--use_attention', type=str2bool, default=False, choices=[True, False], help='Applying block-wise attention for feature processor')
    parser.add_argument('--chans', type=int, default=32, help='Number of channels for cascade U-Net | 18 in original varnet')
    parser.add_argument('--sens-chans', type=int, default=8, help='Number of channels for sensitivity map U-Net | 8 in original varnet')

    # saving hyperparameter
    parser.add_argument('--result-path', type=Path, default='C://Users/bigse/OneDrive/Desktop/fastMRI-2025-NeuroScope/results', help='Directory of train/val results')
    parser.add_argument('--seed', type=int, default=0, help='Fix random seed')
    parser.add_argument('--report-interval', type=int, default=10, help='Interval for printing training status')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)
    

    task = FastMRI(args)
    # task.print_model()
    # task.count_parameters()
    task.train()

