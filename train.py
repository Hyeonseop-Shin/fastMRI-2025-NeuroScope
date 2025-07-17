import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train, train_moe

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=3e-4, help='Max learning rate (after warmup)')
    parser.add_argument('-n', '--net-name', type=Path, default='fivarnet', help='Name of network')
    
    # scheduler hyperparameter
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--anneal1', type=int, default=40, help='First cosine annealing epochs')
    parser.add_argument('--warmup2', type=int, default=10, help='Second warmup epochs')
    parser.add_argument('--anneal2', type=int, default=40, help='Second cosine annealing epochs')
    parser.add_argument('--lr-min1', type=float, default=0.00005, help='Min LR after first annealing')
    parser.add_argument('--lr-max2', type=float, default=0.00015, help='Max LR for second warmup')
    parser.add_argument('--lr-min2', type=float, default=0.0, help='Min LR after second annealing')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine'], help='LR scheduler type')
    
    # Data hyperparameter
    parser.add_argument('-d', '--data-augmentation', type=bool, default=False, help='Apply spatial augmentation')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/root/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/root/Data/val/', help='Directory of validation data')
    
    # MoE hyperparameter (추가!)
    parser.add_argument('--use-moe', type=bool, default=False, help='Use Mixture of Experts training')
    
    # important hyperparameter
    parser.add_argument('-m', '--model', type=str, default='fivarnet', choices=['varnet', 'fivarnet'], help='Model type')
    parser.add_argument('-f', '--feature_cascades', type=int, default=3, help='Number of cascades | Should be less than 12')
    parser.add_argument('-i', '--image_cascades', type=int, default=3, help='Number of cascades | Should be less than 12')
    parser.add_argument('-a', '--use_attention', type=bool, default=True, choices=[True, False], help='Applying block-wise attention for feature processor')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet')
    parser.add_argument('--sens-chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet')
    
    parser.add_argument('--result-path', type=Path, default='/root/FastMRI_challenge/results', help='Directory of train/val results')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=0, help='Fix random seed')
    parser.add_argument('--accumulation-step', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--report-interval', type=int, default=10, help='Interval for printing training status')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)
    
    args.use_moe = True
    args.data_augmentation = True

    # MoE 파이프라인 실행
    if args.use_moe:
        print("========== Starting MoE Training Pipeline ==========")
        print(f"MoE Augmentation: {'Enabled' if args.data_augmentation else 'Disabled'}")
        
        train_moe(args)
        
    else:
        print("========== Starting Standard Training ==========")
        print(f"Data Augmentation: {'Enabled' if args.data_augmentation else 'Disabled'}")
        
        # 표준 훈련용 디렉토리 설정
        args.exp_dir = Path(os.path.join(args.result_path, args.net_name, 'checkpoints'))
        args.val_dir = Path(os.path.join(args.result_path, args.net_name, 'reconstructions_val'))
        args.main_dir = Path(os.path.join(args.result_path, args.net_name, 'main'))
        args.val_loss_dir = Path(os.path.join(args.result_path, args.net_name))

        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.val_dir.mkdir(parents=True, exist_ok=True)

        train(args)
